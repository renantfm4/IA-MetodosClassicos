# -*- coding: utf-8 -*-
"""PROJETO 6 - BASE DE CONHECIMENTOS (Métodos Clássicos de IA)

=====================================================
Problema: Mundo de Wumpus (4x4). Um agente parte da célula (1,1) em uma
caverna com poços (fatais), um Wumpus (fatal) e um ouro. Ele não enxerga a
caverna: só percebe BRISA (há poço em célula vizinha), FEDOR (o Wumpus está
em célula vizinha) e BRILHO (o ouro está na célula atual). O objetivo é pegar
o ouro e voltar à saída (1,1) sem morrer, RACIOCINANDO sobre o que é seguro.

Base de conhecimentos (lógica proposicional, forma normal conjuntiva):
  - símbolos P(x,y) "há poço em (x,y)" e W(x,y) "há Wumpus em (x,y)"
  - cada percepção vira cláusulas (ex.: brisa em (1,2) => P(1,1)v P(2,2)v P(1,3);
    sem brisa => ¬P para todos os vizinhos)
  - axioma opcional: existe exatamente um Wumpus

Consulta: ASK(KB |= alfa) -- a célula é PROVADAMENTE segura?
Motores de inferência implementados (e comparados):
  - TT  : verificação de modelos por tabela-verdade (vetorizada)
  - RES : resolução por refutação (estratégia de conjunto de suporte)
  - DPLL: satisfatibilidade com propagação unitária (usado pelo agente)

Políticas de agente:
  - cauteloso : só visita células provadamente seguras; sem nenhuma, desiste
  - arriscado : sem células seguras, tenta uma célula não provada perigosa
  - aleatório : sem base de conhecimentos (referência)

Como executar:
    python base_conhecimento.py          (leva cerca de 1 a 3 minutos)

Saídas (pasta ./resultados):
  - 6-fig1_mundo.png
  - 6-fig2_inferencia.png
  - 6-fig3_agentes.png
  - resultados-conhecimento.json
  - resultados-conhecimento.txt

Dependências: numpy, matplotlib (o restante é biblioteca padrão do Python).
"""

import itertools
import json
import os
import random
import time
from collections import deque

import matplotlib

matplotlib.use("Agg")  # backend sem janela (salva direto em arquivo)
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position

SAIDA = "resultados"
os.makedirs(SAIDA, exist_ok=True)
SEMENTE = 42
LADO = 4
P_POCO = 0.2
LIMITE_TT = 20  # tabela-verdade só é executada com até 2^20 modelos
LIMITE_RES = 5000  # teto de cláusulas geradas pela resolução


# =====================================================================
# PARTE 1 - SÍMBOLOS, MUNDO E PERCEPÇÕES
# =====================================================================
# Cada célula (r, c), com r = linha (0 embaixo) e c = coluna, recebe dois
# símbolos proposicionais numerados: P (poço) de 1 a 16 e W (Wumpus) de 17 a
# 32. Um literal é o número do símbolo (positivo) ou o seu negativo.
def sim_p(r, c):
    """Símbolo 'há poço em (r,c)'."""
    return LADO * r + c + 1


def sim_w(r, c):
    """Símbolo 'há Wumpus em (r,c)'."""
    return LADO * r + c + 1 + LADO * LADO


def nome_simbolo(s):
    """Nome legível de um símbolo, ex.: 'P(2,3)' (coordenadas a partir de 1)."""
    tipo = "P" if s <= LADO * LADO else "W"
    i = (s - 1) % (LADO * LADO)
    return f"{tipo}({i // LADO + 1},{i % LADO + 1})"


def vizinhas(r, c):
    """Células vizinhas (4 direções) dentro da caverna."""
    return [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if 0 <= r + dr < LADO and 0 <= c + dc < LADO]


def gerar_mundo(rng):
    """Poços com probabilidade P_POCO (exceto a saída); Wumpus e ouro em
    células distintas, sem poço e diferentes da saída."""
    celulas = [(r, c) for r in range(LADO) for c in range(LADO) if (r, c) != (0, 0)]
    while True:
        pocos = {x for x in celulas if rng.random() < P_POCO}
        livres = [x for x in celulas if x not in pocos]
        if len(livres) >= 2:
            wumpus, ouro = rng.sample(livres, 2)
            return dict(pocos=pocos, wumpus=wumpus, ouro=ouro)


def perceber(mundo, pos):
    """Percepções (brisa, fedor, brilho) na célula pos."""
    viz = vizinhas(*pos)
    return (any(v in mundo["pocos"] for v in viz),
            any(v == mundo["wumpus"] for v in viz),
            pos == mundo["ouro"])


def perigosa(mundo, pos):
    """A célula contém poço ou Wumpus?"""
    return pos in mundo["pocos"] or pos == mundo["wumpus"]


def solucionavel(mundo):
    """Existe caminho de (0,0) até o ouro só por células sem perigo?"""
    visto, fila = {(0, 0)}, deque([(0, 0)])
    while fila:
        atual = fila.popleft()
        if atual == mundo["ouro"]:
            return True
        for v in vizinhas(*atual):
            if v not in visto and not perigosa(mundo, v):
                visto.add(v)
                fila.append(v)
    return False


# =====================================================================
# PARTE 2 - BASE DE CONHECIMENTOS (cláusulas em FNC)
# =====================================================================
class BaseConhecimento:
    """Conjunto de cláusulas (frozensets de literais) com TELL de percepções."""

    def __init__(self, unico_wumpus):
        self.clausulas = set()
        self.tell([-sim_p(0, 0)])  # a saída não tem poço
        self.tell([-sim_w(0, 0)])  # nem Wumpus
        if unico_wumpus:  # axioma: existe exatamente um Wumpus
            ws = [sim_w(r, c) for r in range(LADO) for c in range(LADO)
                  if (r, c) != (0, 0)]
            self.tell(ws)  # pelo menos um
            for a, b in itertools.combinations(ws, 2):
                self.tell([-a, -b])  # no máximo um
        self._cache = None

    def tell(self, literais):
        """Adiciona uma cláusula à base."""
        self.clausulas.add(frozenset(literais))
        self._cache = None

    def tell_percepcao(self, pos, brisa, fedor):
        """Converte as percepções de uma célula visitada em cláusulas.
        B(x,y) <=> P(vizinha1) v P(vizinha2) ... (e idem para fedor/Wumpus)."""
        r, c = pos
        self.tell([-sim_p(r, c)])  # estou vivo: sem poço aqui
        self.tell([-sim_w(r, c)])  # nem Wumpus
        viz = vizinhas(r, c)
        for f_sim, perc in ((sim_p, brisa), (sim_w, fedor)):
            if perc:
                self.tell([f_sim(*v) for v in viz])
            else:
                for v in viz:
                    self.tell([-f_sim(*v)])

    def reduzida(self):
        """Base após propagação unitária (com cache): (cláusulas, literais
        provados, consistente)."""
        if self._cache is None:
            self._cache = propagar(list(self.clausulas))
        return self._cache


# =====================================================================
# PARTE 3 - MOTORES DE INFERÊNCIA
# =====================================================================
def propagar(clausulas):
    """Propagação unitária: enquanto houver cláusula unitária {l}, atribui l,
    remove as cláusulas satisfeitas e apaga ¬l das demais. Retorna
    (cláusulas restantes, literais atribuídos, consistente)."""
    cls = list(clausulas)
    atrib = set()
    while True:
        unit = None
        for c in cls:
            if not c:
                return cls, atrib, False  # cláusula vazia: contradição
            if len(c) == 1:
                unit = next(iter(c))
                break
        if unit is None:
            return cls, atrib, True
        atrib.add(unit)
        cls = [c - {-unit} if -unit in c else c for c in cls if unit not in c]


def dpll_sat(clausulas, st):
    """DPLL: a lista de cláusulas é satisfatível? st['chamadas'] conta as
    chamadas recursivas."""
    st["chamadas"] += 1
    cls, _, ok = propagar(clausulas)
    if not ok:
        return False
    if not cls:
        return True
    lit = next(iter(min(cls, key=len)))  # literal da menor cláusula
    return (dpll_sat(cls + [frozenset([lit])], st)
            or dpll_sat(cls + [frozenset([-lit])], st))


def ask_dpll(clausulas, alfa):
    """KB |= alfa  <=>  KB ^ ¬alfa é insatisfatível. Retorna (bool, info)."""
    st = dict(chamadas=0)
    resp = not dpll_sat(list(clausulas) + [frozenset([-alfa])], st)
    return resp, st


def ask_dpll_reduzida(kb, alfa):
    """Versão usada pelo agente: aproveita a base já reduzida por propagação
    unitária (recalculada só quando a base muda)."""
    cls, atrib, ok = kb.reduzida()
    if not ok or alfa in atrib:
        return True
    if -alfa in atrib:
        return False
    return not dpll_sat(cls + [frozenset([-alfa])], dict(chamadas=0))


def ask_tt(clausulas, alfa):
    """Verificação de modelos (tabela-verdade): enumera todas as atribuições
    dos símbolos que sobram após a propagação unitária e confere se alguma
    satisfaz KB ^ ¬alfa. Retorna (bool|None, info); None se houver mais de
    LIMITE_TT símbolos (2^k modelos inviável)."""
    cls, _, ok = propagar(list(clausulas) + [frozenset([-alfa])])
    if not ok:
        return True, dict(k=0)
    simbolos = sorted({abs(literal) for c in cls for literal in c})
    k = len(simbolos)
    if k > LIMITE_TT:
        return None, dict(k=k)
    if k == 0:
        return False, dict(k=0)
    modelos = np.arange(1 << k, dtype=np.int32)
    pos = {s: i for i, s in enumerate(simbolos)}
    satisfaz = np.ones(1 << k, dtype=bool)
    for c in cls:
        alguma = np.zeros(1 << k, dtype=bool)
        for literal in c:
            bit = ((modelos >> pos[abs(literal)]) & 1) == 1
            alguma |= bit if literal > 0 else ~bit
        satisfaz &= alguma
    return not satisfaz.any(), dict(k=k)


def resolver(a, b):
    """Resolventes de duas cláusulas (sem tautologias)."""
    saida = []
    for literal in a:
        if -literal in b:
            r = (a | b) - {literal, -literal}
            if not any(-x in r for x in r):
                saida.append(frozenset(r))
    return saida


def ask_resolucao(clausulas, alfa):
    """Resolução por refutação com conjunto de suporte: parte de ¬alfa e só
    resolve cláusulas que descendem dele. Retorna (True, info) se derivar a
    cláusula vazia, (False, info) se saturar e (None, info) se estourar o
    limite de cláusulas."""
    usaveis = list(clausulas)
    vistos = set(usaveis)
    neg = frozenset([-alfa])
    sos = deque([neg])
    vistos.add(neg)
    gerados = 0
    while sos:
        c = sos.popleft()
        for d in list(usaveis):
            for r in resolver(c, d):
                if not r:
                    return True, dict(gerados=gerados)
                if r not in vistos:
                    vistos.add(r)
                    sos.append(r)
                    gerados += 1
                    if gerados > LIMITE_RES:
                        return None, dict(gerados=gerados)
        usaveis.append(c)
    return False, dict(gerados=gerados)


# =====================================================================
# PARTE 4 - AGENTES
# =====================================================================
def caminho_ate(origem, destinos, permitidas):
    """Menor caminho (BFS) de origem até a célula mais próxima de 'destinos',
    andando só por células 'permitidas'. Retorna lista de células ou None."""
    pai = {origem: None}
    fila = deque([origem])
    while fila:
        atual = fila.popleft()
        if atual in destinos and atual != origem:
            rota = []
            while atual is not None:
                rota.append(atual)
                atual = pai[atual]
            return rota[::-1]
        for v in vizinhas(*atual):
            if v in permitidas and v not in pai:
                pai[v] = atual
                fila.append(v)
    return None


def jogar_kb(mundo, politica, unico, rng, registro=None, max_passos=100):
    """Agente baseado em conhecimento. Retorna dict com resultado ('vitoria',
    'morte' ou 'desistiu'), passos, consultas ao KB, pontuação (+1000 ouro
    na saída, -1000 morte, -1 por passo) e o conhecimento final."""
    kb = BaseConhecimento(unico)
    pos = (0, 0)
    visitadas, seguras, perigosas = {pos}, {pos}, set()
    percepcoes, trilha = {}, [pos]
    passos = consultas = falsos_seguros = 0
    ouro = False
    resultado = "desistiu"

    def mover(destino):
        nonlocal pos, passos
        pos = destino
        passos += 1
        trilha.append(pos)

    def perguntar(literal):
        nonlocal consultas
        consultas += 1
        if registro is not None:
            registro.append((list(kb.clausulas), literal))
        return ask_dpll_reduzida(kb, literal)

    while passos < max_passos:
        brisa, fedor, brilho = perceber(mundo, pos)
        percepcoes[pos] = (brisa, fedor)
        kb.tell_percepcao(pos, brisa, fedor)
        if brilho:
            ouro = True
            break
        # --- raciocínio: classificar as células da fronteira ---
        fronteira = {v for p in visitadas for v in vizinhas(*p)} - visitadas
        for cel in fronteira - seguras - perigosas:
            if perguntar(-sim_p(*cel)) and perguntar(-sim_w(*cel)):
                seguras.add(cel)
                falsos_seguros += perigosa(mundo, cel)  # verificação de correção
            elif perguntar(sim_p(*cel)) or perguntar(sim_w(*cel)):
                perigosas.add(cel)
        # --- decisão ---
        alvo = caminho_ate(pos, seguras - visitadas, seguras)
        if alvo is None and politica == "arriscado":
            incertas = fronteira - perigosas - seguras
            if incertas:
                cel = rng.choice(sorted(incertas))
                ponte = caminho_ate(pos, {v for v in vizinhas(*cel) if v in visitadas},
                                    visitadas) if pos not in vizinhas(*cel) else [pos]
                if ponte is not None:
                    alvo = ponte + [cel] if ponte[-1] != cel else ponte
                    if alvo[0] == pos and len(alvo) > 1 and alvo[1] == pos:
                        alvo = alvo[1:]
        if alvo is None:
            break  # desiste
        for celula in alvo[1:]:
            mover(celula)
            if perigosa(mundo, pos):
                resultado = "morte"
                break
        if resultado == "morte":
            break
        visitadas.add(pos)
    if resultado != "morte" and ouro:
        volta = caminho_ate(pos, {(0, 0)}, visitadas) if pos != (0, 0) else [pos]
        passos += len(volta) - 1
        trilha.extend(volta[1:])
        resultado = "vitoria"
    elif resultado != "morte":
        volta = caminho_ate(pos, {(0, 0)}, visitadas) if pos != (0, 0) else [pos]
        passos += len(volta) - 1 if volta else 0
        resultado = "desistiu"
    pontos = -passos + (1000 if resultado == "vitoria" else 0) \
        - (1000 if resultado == "morte" else 0)
    return dict(resultado=resultado, passos=passos, consultas=consultas,
                pontos=pontos, falsos_seguros=falsos_seguros, ouro=ouro,
                visitadas=visitadas, seguras=seguras, perigosas=perigosas,
                percepcoes=percepcoes, trilha=trilha)


def jogar_aleatorio(mundo, rng, max_passos=100):
    """Referência sem conhecimento: passeio aleatório até achar o ouro,
    morrer ou esgotar os passos."""
    pos, passos = (0, 0), 0
    while passos < max_passos:
        pos = rng.choice(vizinhas(*pos))
        passos += 1
        if perigosa(mundo, pos):
            return dict(resultado="morte", passos=passos, ouro=False)
        if pos == mundo["ouro"]:
            return dict(resultado="ouro", passos=passos, ouro=True)
    return dict(resultado="desistiu", passos=passos, ouro=False)


# =====================================================================
# PARTE 5 - FIGURAS
# =====================================================================
def desenhar_caverna(ax, mundo, conhecimento, titulo):
    """Desenha a caverna. Sem 'conhecimento': o mundo real. Com
    'conhecimento': o que o agente sabe (visitada, segura, perigosa, ?)."""
    ax.set_xlim(0, LADO)
    ax.set_ylim(0, LADO)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for r in range(LADO):
        for c in range(LADO):
            cel = (r, c)
            cor, txt = "white", ""
            if conhecimento is None:
                if cel in mundo["pocos"]:
                    cor, txt = "#555555", "Poço"
                elif cel == mundo["wumpus"]:
                    cor, txt = "#e57373", "Wumpus"
                elif cel == mundo["ouro"]:
                    cor, txt = "#ffd54f", "Ouro"
                elif cel == (0, 0):
                    txt = "Saída"
            else:
                if cel in conhecimento["visitadas"]:
                    cor = "#9ad19a"
                    b, f = conhecimento["percepcoes"].get(cel, (False, False))
                    txt = ("B" if b else "") + ("F" if f else "")
                elif cel in conhecimento["seguras"]:
                    cor = "#d8f0d8"
                elif cel in conhecimento["perigosas"]:
                    cor, txt = "#f4a6a6", "perigo"
                else:
                    cor, txt = "#dddddd", "?"
            ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=cor,
                                       edgecolor="black", lw=1))
            ax.text(c + 0.5, r + 0.5, txt, ha="center", va="center",
                    fontsize=8, color="white" if cor == "#555555" else "black")
    if conhecimento is not None:
        xs = [c + 0.5 for _, c in conhecimento["trilha"]]
        ys = [r + 0.5 for r, _ in conhecimento["trilha"]]
        ax.plot(xs, ys, "-", color="#1f6fb2", lw=1.8)
        ax.plot(xs[0], ys[0], "o", color="#1f6fb2", ms=6)
    ax.set_title(titulo, fontsize=9)


# =====================================================================
# PARTE 6 - EXPERIMENTOS
# =====================================================================
def main():
    """Executa os experimentos, gera as figuras e salva os resultados."""
    todos = {}
    linhas = []

    def log(msg=""):
        print(msg)
        linhas.append(msg)

    rng = random.Random(SEMENTE)
    n_mundos = 300
    mundos = [gerar_mundo(rng) for _ in range(n_mundos)]
    solv = [solucionavel(m) for m in mundos]
    log(f"=== {n_mundos} mundos {LADO}x{LADO} (poço com prob. {P_POCO}) ===")
    log(f"Mundos solucionáveis (ouro alcançável sem passar por perigo): "
        f"{sum(solv)} ({100*sum(solv)/n_mundos:.1f}%)")
    todos["mundos"] = dict(total=n_mundos, solucionaveis=int(sum(solv)))

    # ---------- Experimento 1: um mundo de exemplo ----------
    exemplo = None
    for m in mundos:
        if solucionavel(m) and len(m["pocos"]) >= 2:
            r = jogar_kb(m, "cauteloso", True, random.Random(SEMENTE))
            if r["resultado"] == "vitoria" and r["passos"] >= 12:
                exemplo, res_ex = m, r
                break
    log("\n=== Experimento 1: mundo de exemplo (agente cauteloso, KB com unicidade) ===")
    log(f"Poços: {sorted((r + 1, c + 1) for r, c in exemplo['pocos'])} | Wumpus: "
        f"{(exemplo['wumpus'][0] + 1, exemplo['wumpus'][1] + 1)} | Ouro: "
        f"{(exemplo['ouro'][0] + 1, exemplo['ouro'][1] + 1)}")
    log(f"Resultado: {res_ex['resultado']} | passos={res_ex['passos']} | "
        f"consultas ao KB={res_ex['consultas']} | pontos={res_ex['pontos']}")
    log(f"Células visitadas={len(res_ex['visitadas'])} | provadas seguras="
        f"{len(res_ex['seguras'])} | provadas perigosas={len(res_ex['perigosas'])}")
    todos["exemplo"] = dict(resultado=res_ex["resultado"], passos=res_ex["passos"],
                            consultas=res_ex["consultas"], pontos=res_ex["pontos"],
                            visitadas=len(res_ex["visitadas"]),
                            seguras=len(res_ex["seguras"]),
                            perigosas=len(res_ex["perigosas"]))
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 4.2))
    desenhar_caverna(axs[0], exemplo, None, "Mundo real (oculto ao agente)")
    desenhar_caverna(axs[1], exemplo, res_ex,
                     "Conhecimento do agente (B = brisa, F = fedor)")
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.savefig(f"{SAIDA}/6-fig1_mundo.png", dpi=170)
    plt.close()

    # ---------- Experimento 2: comparação dos motores de inferência ----------
    log("\n=== Experimento 2: motores de inferência (mesmas consultas, KB sem axioma de unicidade) ===")
    consultas = []
    for m in mundos[:40]:
        jogar_kb(m, "cauteloso", False, random.Random(SEMENTE), registro=consultas)
    log(f"Consultas coletadas: {len(consultas)}")
    faixas = [(0, 4), (5, 8), (9, 12), (13, 16), (17, 20)]
    dados = {f: dict(n=0, tt=[], res=[], dpll=[], tt_pulou=0, res_pulou=0)
             for f in faixas}
    discordancias = comparadas = 0
    for cls, alfa in consultas:
        t0 = time.perf_counter()
        r_dpll, _ = ask_dpll(cls, alfa)
        t_dpll = time.perf_counter() - t0
        t0 = time.perf_counter()
        r_res, _ = ask_resolucao(cls, alfa)
        t_res = time.perf_counter() - t0
        t0 = time.perf_counter()
        r_tt, info = ask_tt(cls, alfa)
        t_tt = time.perf_counter() - t0
        k = info["k"]
        faixa = next((f for f in faixas if f[0] <= k <= f[1]), None)
        if faixa is None:
            continue
        d = dados[faixa]
        d["n"] += 1
        d["dpll"].append(t_dpll * 1000)
        if r_res is None:
            d["res_pulou"] += 1
        else:
            d["res"].append(t_res * 1000)
        if r_tt is None:
            d["tt_pulou"] += 1
        else:
            d["tt"].append(t_tt * 1000)
        for r in (r_res, r_tt):
            if r is not None:
                comparadas += 1
                discordancias += int(r != r_dpll)
    todos["inferencia"] = {}
    log(f"Respostas comparadas: {comparadas} | discordâncias com o DPLL: {discordancias}")
    log(f"{'Símbolos':>9s} {'consultas':>9s} | {'TT (ms)':>9s} | {'RES (ms)':>9s} | {'DPLL (ms)':>9s}")
    for f, d in dados.items():
        if d["n"] == 0:
            continue
        med = {k: (float(np.mean(d[k])) if d[k] else None) for k in ("tt", "res", "dpll")}
        todos["inferencia"][f"{f[0]}-{f[1]}"] = dict(
            consultas=d["n"], tt_ms=med["tt"], res_ms=med["res"], dpll_ms=med["dpll"],
            tt_puladas=d["tt_pulou"], res_puladas=d["res_pulou"])
        fmt = lambda v: f"{v:9.3f}" if v is not None else "      n/d"
        log(f"{f[0]:>3d}-{f[1]:<5d} {d['n']:9d} | {fmt(med['tt'])} | {fmt(med['res'])} | {fmt(med['dpll'])}")
    todos["inferencia_resumo"] = dict(consultas=len(consultas), comparadas=comparadas,
                                      discordancias=discordancias)
    full = BaseConhecimento(True)
    log(f"KB completa com unicidade: {len(full.clausulas)} cláusulas e "
        f"{2 * LADO * LADO - 2} símbolos livres -> tabela-verdade teria 2^30 ≈ 1,07 bilhão de modelos")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    cores = {"tt": "#D62728", "res": "#F58518", "dpll": "#54A24B"}
    nomes = {"tt": "Tabela-verdade", "res": "Resolução", "dpll": "DPLL"}
    for chave in ("tt", "res", "dpll"):
        xs, ys = [], []
        for i, f in enumerate(faixas):
            v = todos["inferencia"].get(f"{f[0]}-{f[1]}", {}).get(chave + "_ms")
            if v is not None:
                xs.append(i)
                ys.append(v)
        ax.plot(xs, ys, "o-", color=cores[chave], label=nomes[chave])
    ax.set_yscale("log")
    ax.set_xticks(range(len(faixas)))
    ax.set_xticklabels([f"{f[0]}-{f[1]}\n(n={dados[f]['n']})" for f in faixas], fontsize=8)
    ax.set_xlabel("Símbolos livres após propagação unitária (e nº de consultas)")
    ax.set_ylabel("Tempo médio por consulta (ms)")
    ax.set_title("Custo de inferência por tamanho do problema", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{SAIDA}/6-fig2_inferencia.png", dpi=170)
    plt.close()

    # ---------- Experimento 3: agentes em 300 mundos ----------
    log("\n=== Experimento 3: agentes nos 300 mundos ===")
    politicas = {
        "KB cauteloso (c/ unicidade)": ("cauteloso", True),
        "KB cauteloso (sem unicidade)": ("cauteloso", False),
        "KB arriscado (c/ unicidade)": ("arriscado", True),
    }
    todos["agentes"] = {}
    for nome, (pol, unico) in politicas.items():
        rs = [jogar_kb(m, pol, unico, random.Random(SEMENTE + k))
              for k, m in enumerate(mundos)]
        cont = {x: sum(1 for r in rs if r["resultado"] == x)
                for x in ("vitoria", "desistiu", "morte")}
        vit_solv = sum(1 for r, s in zip(rs, solv) if s and r["resultado"] == "vitoria")
        d = dict(vitoria=100 * cont["vitoria"] / n_mundos,
                 desistiu=100 * cont["desistiu"] / n_mundos,
                 morte=100 * cont["morte"] / n_mundos,
                 vitoria_entre_solucionaveis=100 * vit_solv / sum(solv),
                 pontos=float(np.mean([r["pontos"] for r in rs])),
                 passos=float(np.mean([r["passos"] for r in rs])),
                 consultas=float(np.mean([r["consultas"] for r in rs])),
                 falsos_seguros=int(sum(r["falsos_seguros"] for r in rs)))
        todos["agentes"][nome] = d
        log(f"{nome:30s} vitória={d['vitoria']:5.1f}% (solucionáveis: "
            f"{d['vitoria_entre_solucionaveis']:5.1f}%) desistiu={d['desistiu']:5.1f}% "
            f"morte={d['morte']:5.1f}% pontos={d['pontos']:7.1f} passos={d['passos']:5.1f} "
            f"consultas={d['consultas']:6.1f} falsos seguros={d['falsos_seguros']}")
    rs = [jogar_aleatorio(m, random.Random(SEMENTE + k)) for k, m in enumerate(mundos)]
    cont = {x: sum(1 for r in rs if r["resultado"] == x) for x in ("ouro", "desistiu", "morte")}
    d = dict(ouro=100 * cont["ouro"] / n_mundos, desistiu=100 * cont["desistiu"] / n_mundos,
             morte=100 * cont["morte"] / n_mundos,
             passos=float(np.mean([r["passos"] for r in rs])))
    todos["agentes"]["Aleatório (sem KB)"] = d
    log(f"{'Aleatório (sem KB)':30s} chegou ao ouro={d['ouro']:5.1f}% (sem voltar) "
        f"desistiu={d['desistiu']:5.1f}% morte={d['morte']:5.1f}% passos={d['passos']:5.1f}")

    nomes_ag = list(todos["agentes"])
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    base = np.zeros(len(nomes_ag))
    for chave, cor, rot in (("vitoria", "#54A24B", "Vitória (ouro + saída)"),
                            ("ouro", "#B5D99C", "Ouro sem voltar (aleatório)"),
                            ("desistiu", "#BBBBBB", "Desistiu / sem ouro"),
                            ("morte", "#D62728", "Morreu")):
        vals = np.array([todos["agentes"][n].get(chave, 0) for n in nomes_ag])
        axs[0].barh(range(len(nomes_ag)), vals, left=base, color=cor, label=rot)
        base += vals
    axs[0].set_yticks(range(len(nomes_ag)))
    axs[0].set_yticklabels(nomes_ag, fontsize=7)
    axs[0].set_xlabel("% dos 300 mundos")
    axs[0].set_title("Resultado das partidas", fontsize=10)
    axs[0].legend(fontsize=6, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    axs[0].invert_yaxis()
    kb_nomes = [n for n in nomes_ag if n.startswith("KB")]
    axs[1].bar(range(len(kb_nomes)), [todos["agentes"][n]["pontos"] for n in kb_nomes],
               color="#4C78A8")
    for i, n in enumerate(kb_nomes):
        axs[1].text(i, todos["agentes"][n]["pontos"] + 8,
                    f"{todos['agentes'][n]['pontos']:.0f}", ha="center", fontsize=8)
    axs[1].set_xticks(range(len(kb_nomes)))
    axs[1].set_xticklabels([n.replace(" (", "\n(") for n in kb_nomes], fontsize=7)
    axs[1].set_ylabel("Pontuação média")
    axs[1].set_title("Pontuação média (+1000 ouro, -1000 morte, -1/passo)", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{SAIDA}/6-fig3_agentes.png", dpi=170)
    plt.close()

    # ---------- Arquivos de saída ----------
    with open(f"{SAIDA}/resultados-conhecimento.json", "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)
    with open(f"{SAIDA}/resultados-conhecimento.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("RELATÓRIO DE RESULTADOS - PROJETO 6: BASE DE CONHECIMENTOS\n")
        f.write("=" * 70 + "\n\n" + "\n".join(linhas) + "\n")
    print(f"\nArquivos salvos com sucesso na pasta ./{SAIDA}:\n"
          "  - 6-fig1_mundo.png\n"
          "  - 6-fig2_inferencia.png\n"
          "  - 6-fig3_agentes.png\n"
          "  - resultados-conhecimento.json\n"
          "  - resultados-conhecimento.txt\n")


if __name__ == "__main__":
    main()