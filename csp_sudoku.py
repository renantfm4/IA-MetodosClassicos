# -*- coding: utf-8 -*-
"""PROJETO 5 - PROBLEMAS DE SATISFAÇÃO DE RESTRIÇÕES / CSP (Métodos Clássicos de IA)

=====================================================
Problema: Sudoku modelado como CSP.
  - Variáveis  : as 81 células do tabuleiro
  - Domínios   : {1, ..., 9} (células dadas têm domínio unitário)
  - Restrições : em cada linha, coluna e bloco 3x3 os valores são todos
                 diferentes (27 restrições AllDiff = 810 restrições binárias
                 do tipo "diferente de" entre pares de células vizinhas)

Métodos comparados:
  - BT           : retrocesso (backtracking) simples
  - FC           : retrocesso + verificação adiante (forward checking)
  - FC+MRV       : FC + heurística de variável mais restrita (MRV, desempate
                   por grau)
  - MAC+MRV      : manutenção de consistência de arco (AC-3 a cada atribuição)
  - MAC+AllDiff  : MAC + propagação da restrição global AllDiff ("únicos
                   ocultos": se um dígito só cabe em uma célula da unidade)
  - Min-conflitos: busca local (não sistemática) para CSP

Como executar:
    python csp_sudoku.py          (leva cerca de 1 a 3 minutos)

Saídas (pasta ./resultados):
  - 5-fig1_sudoku.png
  - 5-fig2_desempenho.png
  - 5-fig3_propagacao.png
  - resultados-csp.json
  - resultados-csp.txt

Dependências: numpy, matplotlib (o restante é biblioteca padrão do Python).
"""

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
LIMITE_NOS = 300_000  # teto de atribuições por resolução (evita travar o BT)

# =====================================================================
# PARTE 1 - MODELAGEM DO CSP (estruturas fixas do Sudoku)
# =====================================================================
# Células indexadas de 0 a 80 (índice = 9*linha + coluna).
# Domínios são máscaras de 9 bits: o bit (d-1) ligado significa "d possível".
TODOS = 0b111111111
UNIDADES = (
    [[9 * r + c for c in range(9)] for r in range(9)]  # 9 linhas
    + [[9 * r + c for r in range(9)] for c in range(9)]  # 9 colunas
    + [[9 * (3 * br + r) + 3 * bc + c for r in range(3) for c in range(3)]
       for br in range(3) for bc in range(3)]  # 9 blocos 3x3
)
VIZ = [sorted({p for un in UNIDADES if i in un for p in un} - {i})
       for i in range(81)]  # os 20 vizinhos de cada célula
POP = [bin(x).count("1") for x in range(512)]  # tamanho do domínio


def bits(mascara):
    """Lista de dígitos (1..9) presentes em uma máscara de domínio."""
    return [d for d in range(1, 10) if mascara >> (d - 1) & 1]


# =====================================================================
# PARTE 2 - PROPAGAÇÃO DE RESTRIÇÕES
# =====================================================================
def ac3(dom, fila):
    """AC-3: processa arcos (xi, xj) da fila. Para a restrição xi != xj, só
    há poda quando D(xj) é unitário: o valor é removido de D(xi). Se D(xi)
    mudar, os arcos (xk, xi) dos vizinhos de xi voltam à fila (enfileirados
    apenas se D(xi) ficou unitário, único caso em que podem podar).
    Retorna False se algum domínio ficar vazio."""
    while fila:
        xi, xj = fila.popleft()
        dj = dom[xj]
        if dj and dj & (dj - 1) == 0 and dom[xi] & dj:
            dom[xi] &= ~dj
            if not dom[xi]:
                return False
            if dom[xi] & (dom[xi] - 1) == 0:  # ficou unitário
                for xk in VIZ[xi]:
                    if xk != xj:
                        fila.append((xk, xi))
    return True


def alldiff(dom):
    """Propagação da restrição global AllDiff por "únicos ocultos": se um
    dígito só pode ir em uma célula de uma unidade, essa célula recebe o
    dígito; se algum dígito não cabe em nenhuma célula, há falha."""
    mudou = True
    while mudou:
        mudou = False
        for un in UNIDADES:
            uma, duas = 0, 0
            for c in un:
                duas |= uma & dom[c]
                uma |= dom[c]
            if uma != TODOS:
                return False  # algum dígito não cabe em nenhuma célula
            for d in range(9):
                bit = 1 << d
                if uma & ~duas & bit:  # dígito aparece em exatamente 1 célula
                    # (reconfere, pois a propagação pode ter mudado os domínios)
                    cands = [c for c in un if dom[c] & bit]
                    if not cands:
                        return False
                    if len(cands) != 1:
                        continue
                    c = cands[0]
                    if dom[c] != bit:
                        dom[c] = bit
                        mudou = True
                        fila = deque((p, c) for p in VIZ[c])
                        if not ac3(dom, fila):
                            return False
    return True


def atribuir(dom, val, i, d, modo):
    """Atribui d à variável i e propaga conforme o modo: 'fc' remove d dos
    vizinhos; 'mac' roda AC-3; 'mac_all' roda AC-3 e AllDiff."""
    bit = 1 << (d - 1)
    if not dom[i] & bit:
        return False
    val[i] = d
    dom[i] = bit
    if modo == "fc":
        for p in VIZ[i]:
            if dom[p] & bit:
                dom[p] &= ~bit
                if not dom[p]:
                    return False
        return True
    if not ac3(dom, deque((p, i) for p in VIZ[i])):
        return False
    return alldiff(dom) if modo == "mac_all" else True


def inicializar(grade, modo):
    """Cria os domínios iniciais e propaga as pistas dadas. Retorna
    (dom, val, ok)."""
    dom, val = [TODOS] * 81, [0] * 81
    for i, d in enumerate(grade):
        if d and not atribuir(dom, val, i, d, modo):
            return dom, val, False
    if modo == "mac_all" and not alldiff(dom):
        return dom, val, False
    return dom, val, True


# =====================================================================
# PARTE 3 - ALGORITMOS DE BUSCA
# =====================================================================
def bt_simples(grade, limite_nos=LIMITE_NOS):
    """Retrocesso simples: variáveis na ordem fixa, valores de 1 a 9; só
    verifica consistência com as variáveis já atribuídas (sem propagação)."""
    t0 = time.perf_counter()
    val = list(grade)
    livres = [i for i in range(81) if val[i] == 0]
    st = dict(nos=0, retro=0, estourou=False)

    def rec(k):
        if k == len(livres):
            return True
        i = livres[k]
        for d in range(1, 10):
            if st["nos"] >= limite_nos:
                st["estourou"] = True
                return False
            st["nos"] += 1
            if all(val[p] != d for p in VIZ[i]):
                val[i] = d
                if rec(k + 1):
                    return True
                val[i] = 0
                if st["estourou"]:
                    return False
            st["retro"] += 1
        return False

    achou = rec(0)
    return dict(solucao=val if achou else None, nos=st["nos"],
                retro=st["retro"], estourou=st["estourou"],
                tempo=time.perf_counter() - t0)


def buscar(grade, modo="mac_all", mrv=True, limite_nos=LIMITE_NOS,
           max_sol=1, rng=None):
    """Busca com retrocesso sobre domínios, com propagação ('fc', 'mac' ou
    'mac_all') e, opcionalmente, a heurística MRV (menor domínio; desempate
    pelo maior número de vizinhos ainda não atribuídos). 'nos' conta as
    atribuições tentadas e 'retro' as que não levaram a solução."""
    t0 = time.perf_counter()
    st = dict(nos=0, retro=0, estourou=False)
    solucoes = []
    dom, val, ok = inicializar(grade, modo)

    def escolher(dom, val):
        melhor, melhor_n, melhor_g = -1, 10, -1
        for i in range(81):
            if val[i]:
                continue
            if not mrv:
                return i
            n = POP[dom[i]]
            if n < melhor_n:
                melhor, melhor_n, melhor_g = i, n, -1
            if n == melhor_n and n > 1:  # desempate por grau
                if melhor_g < 0:
                    melhor_g = sum(1 for p in VIZ[melhor] if not val[p])
                g = sum(1 for p in VIZ[i] if not val[p])
                if g > melhor_g:
                    melhor, melhor_g = i, g
        return melhor

    def rec(dom, val):
        i = escolher(dom, val)
        if i < 0:
            solucoes.append(list(val))
            return
        digitos = bits(dom[i])
        if rng is not None:
            rng.shuffle(digitos)
        for d in digitos:
            if st["nos"] >= limite_nos:
                st["estourou"] = True
                return
            st["nos"] += 1
            antes = len(solucoes)
            nd, nv = dom[:], val[:]
            if atribuir(nd, nv, i, d, modo):
                rec(nd, nv)
            if len(solucoes) >= max_sol or st["estourou"]:
                return
            if len(solucoes) == antes:
                st["retro"] += 1

    if ok:
        rec(dom, val)
    return dict(solucao=solucoes[0] if solucoes else None, solucoes=solucoes,
                nos=st["nos"], retro=st["retro"], estourou=st["estourou"],
                tempo=time.perf_counter() - t0)


def min_conflitos(grade, semente, max_passos=20000, ruido=0.02):
    """Busca local (min-conflicts): atribuição completa aleatória das células
    livres; a cada passo escolhe uma variável em conflito e dá a ela o valor
    que minimiza os conflitos (com probabilidade 'ruido', um valor
    aleatório). Não é completo: pode parar sem solução."""
    t0 = time.perf_counter()
    rnd = random.Random(semente)
    val = list(grade)
    fixa = [v != 0 for v in val]
    for i in range(81):
        if not fixa[i]:
            val[i] = rnd.randint(1, 9)
    cnt = [[0] * 10 for _ in range(81)]  # cnt[i][d] = vizinhos de i com valor d
    for i in range(81):
        for p in VIZ[i]:
            cnt[i][val[p]] += 1
    passos = 0
    while passos < max_passos:
        conf = [i for i in range(81) if not fixa[i] and cnt[i][val[i]] > 0]
        if not conf:
            break
        i = rnd.choice(conf)
        if rnd.random() < ruido:
            novo = rnd.randint(1, 9)
        else:
            m = min(cnt[i][1:])
            novo = rnd.choice([d for d in range(1, 10) if cnt[i][d] == m])
        antigo, val[i] = val[i], novo
        for p in VIZ[i]:
            cnt[p][antigo] -= 1
            cnt[p][novo] += 1
        passos += 1
    conf = [i for i in range(81) if not fixa[i] and cnt[i][val[i]] > 0]
    return dict(resolvido=not conf, passos=passos, conflitos=len(conf),
                tempo=time.perf_counter() - t0)


# =====================================================================
# PARTE 4 - GERAÇÃO DE PUZZLES E MÉTRICAS
# =====================================================================
def gerar_puzzle(rng, pistas_alvo):
    """Gera uma solução completa aleatória e remove pistas, mantendo a
    solução ÚNICA, até chegar ao alvo (ou até não ser mais possível)."""
    completa = buscar([0] * 81, "mac_all", True, rng=rng)["solucao"]
    grade = list(completa)
    ordem = list(range(81))
    rng.shuffle(ordem)
    pistas = 81
    for i in ordem:
        if pistas <= pistas_alvo:
            break
        guardado, grade[i] = grade[i], 0
        if len(buscar(grade, "mac_all", True, max_sol=2)["solucoes"]) != 1:
            grade[i] = guardado
        else:
            pistas -= 1
    return grade


NIVEIS = {"Fácil": 38, "Médio": 30, "Difícil": 26, "Muito difícil": 23}
ALGORITMOS = {
    "BT": lambda g: bt_simples(g),
    "FC": lambda g: buscar(g, "fc", False),
    "FC+MRV": lambda g: buscar(g, "fc", True),
    "MAC+MRV": lambda g: buscar(g, "mac", True),
    "MAC+AllDiff": lambda g: buscar(g, "mac_all", True),
}


def valido(grade, solucao):
    """Confere se 'solucao' respeita as pistas e todas as restrições."""
    if solucao is None or any(g and g != s for g, s in zip(grade, solucao)):
        return False
    return all(sorted(solucao[c] for c in un) == list(range(1, 10))
               for un in UNIDADES)


def desenhar_sudoku(ax, grade, solucao, titulo):
    """Desenha o tabuleiro: pistas em preto, valores encontrados em azul."""
    ax.set_xlim(0, 9)
    ax.set_ylim(9, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    for k in range(10):
        lw = 2.2 if k % 3 == 0 else 0.5
        ax.plot([0, 9], [k, k], "k", lw=lw)
        ax.plot([k, k], [0, 9], "k", lw=lw)
    for i in range(81):
        r, c = divmod(i, 9)
        if grade[i]:
            ax.text(c + 0.5, r + 0.55, str(grade[i]), ha="center", va="center",
                    fontsize=12, fontweight="bold")
        elif solucao is not None:
            ax.text(c + 0.5, r + 0.55, str(solucao[i]), ha="center",
                    va="center", fontsize=12, color="#1f6fb2")
    ax.set_title(titulo, fontsize=9)


def desenhar_dominios(ax, dom, titulo):
    """Mapa de calor do tamanho dos domínios (nº de candidatos por célula)."""
    tam = np.array([POP[d] for d in dom]).reshape(9, 9)
    ax.imshow(tam, cmap="Blues", vmin=1, vmax=9)
    for r in range(9):
        for c in range(9):
            ax.text(c, r, str(tam[r, c]), ha="center", va="center", fontsize=8,
                    color="white" if tam[r, c] > 5 else "black")
    for k in (2.5, 5.5):
        ax.axhline(k, color="k", lw=1.5)
        ax.axvline(k, color="k", lw=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    resolvidas = int((tam == 1).sum())
    ax.set_title(f"{titulo}\n{resolvidas}/81 células com 1 candidato", fontsize=8)


# =====================================================================
# PARTE 5 - EXPERIMENTOS
# =====================================================================
def main():
    """Executa os experimentos, gera as figuras e salva os resultados."""
    todos = {}
    linhas = []

    def log(msg=""):
        print(msg)
        linhas.append(msg)

    rng = random.Random(SEMENTE)
    n_por_nivel = 10
    puzzles = {nome: [gerar_puzzle(rng, alvo) for _ in range(n_por_nivel)]
               for nome, alvo in NIVEIS.items()}
    pistas = {nome: float(np.mean([sum(1 for v in g if v) for g in lista]))
              for nome, lista in puzzles.items()}
    log("=== Puzzles gerados (solução única), 10 por nível ===")
    for nome in NIVEIS:
        log(f"{nome:14s} média de pistas = {pistas[nome]:.1f}")
    todos["pistas"] = pistas

    # ---------- Experimento 1: um puzzle "muito difícil" ----------
    exemplo = puzzles["Muito difícil"][0]
    log(f"\n=== Experimento 1: puzzle de exemplo ({sum(1 for v in exemplo if v)} pistas) ===")
    todos["exemplo"] = {}
    solucao_ex = None
    for nome, f in ALGORITMOS.items():
        r = f(exemplo)
        ok = valido(exemplo, r["solucao"])
        solucao_ex = r["solucao"] if ok else solucao_ex
        log(f"{nome:12s} resolvido={str(ok):5s} atribuições={r['nos']:7d} "
            f"retrocessos={r['retro']:7d} t={r['tempo']*1000:9.2f} ms"
            + (" (limite atingido)" if r["estourou"] else ""))
        todos["exemplo"][nome] = dict(resolvido=ok, nos=r["nos"], retro=r["retro"],
                                      tempo_ms=r["tempo"] * 1000)
    mc = min_conflitos(exemplo, SEMENTE)
    log(f"{'Min-conflitos':12s} resolvido={str(mc['resolvido']):5s} passos="
        f"{mc['passos']:7d} conflitos finais={mc['conflitos']} "
        f"t={mc['tempo']*1000:.2f} ms")
    todos["exemplo"]["Min-conflitos"] = dict(
        resolvido=mc["resolvido"], passos=mc["passos"],
        conflitos=mc["conflitos"], tempo_ms=mc["tempo"] * 1000)

    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.8))
    desenhar_sudoku(axs[0], exemplo, None, "Puzzle (pistas em preto)")
    desenhar_sudoku(axs[1], exemplo, solucao_ex, "Solução encontrada (azul)")
    plt.tight_layout()
    plt.savefig(f"{SAIDA}/5-fig1_sudoku.png", dpi=170)
    plt.close()

    # ---------- Experimento 2: 10 puzzles por nível ----------
    log("\n=== Experimento 2: média em 10 puzzles por nível ===")
    todos["niveis"] = {}
    for nome, lista in puzzles.items():
        todos["niveis"][nome] = {}
        log(f"-- {nome} ({pistas[nome]:.1f} pistas) --")
        for alg, f in ALGORITMOS.items():
            rs = [f(g) for g in lista]
            resolvidos = [valido(g, r["solucao"]) for g, r in zip(lista, rs)]
            d = dict(nos=float(np.mean([r["nos"] for r in rs])),
                     retro=float(np.mean([r["retro"] for r in rs])),
                     tempo_ms=float(np.mean([r["tempo"] for r in rs]) * 1000),
                     pct_resolvido=100 * float(np.mean(resolvidos)))
            todos["niveis"][nome][alg] = d
            log(f"{alg:12s} resolvidos={d['pct_resolvido']:5.0f}% "
                f"atribuições={d['nos']:10.1f} retrocessos={d['retro']:10.1f} "
                f"t={d['tempo_ms']:9.2f} ms")
        mcs = [min_conflitos(g, SEMENTE + k) for k, g in enumerate(lista)]
        d = dict(pct_resolvido=100 * float(np.mean([m["resolvido"] for m in mcs])),
                 passos=float(np.mean([m["passos"] for m in mcs])),
                 conflitos=float(np.mean([m["conflitos"] for m in mcs])),
                 tempo_ms=float(np.mean([m["tempo"] for m in mcs]) * 1000))
        todos["niveis"][nome]["Min-conflitos"] = d
        log(f"{'Min-conflitos':12s} resolvidos={d['pct_resolvido']:5.0f}% "
            f"passos={d['passos']:.0f} conflitos finais (média)={d['conflitos']:.1f} "
            f"t={d['tempo_ms']:.2f} ms")

    cores = ["#999999", "#F58518", "#B279A2", "#4C78A8", "#54A24B"]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.7))
    larg = 0.16
    for ax, chave, tit in ((axs[0], "nos", "Atribuições tentadas (média, escala log)"),
                           (axs[1], "tempo_ms", "Tempo por puzzle (ms, escala log)")):
        for k, alg in enumerate(ALGORITMOS):
            vals = [max(todos["niveis"][n][alg][chave], 0.5) for n in NIVEIS]
            ax.bar(np.arange(4) + (k - 2) * larg, vals, larg, label=alg, color=cores[k])
        ax.set_yscale("log")
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"{n}\n({pistas[n]:.0f} pistas)" for n in NIVEIS], fontsize=8)
        ax.set_title(tit, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axs[0].axhline(LIMITE_NOS, color="red", ls=":", lw=1)
    axs[0].text(3.4, LIMITE_NOS * 1.15, "limite", color="red", fontsize=7, ha="right")
    axs[0].legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{SAIDA}/5-fig2_desempenho.png", dpi=170)
    plt.close()

    # ---------- Experimento 3: poder da propagação (sem busca) ----------
    log("\n=== Experimento 3: propagação inicial, sem busca (10 puzzles por nível) ===")
    todos["propagacao"] = {}
    for nome, lista in puzzles.items():
        res = {}
        for modo, rotulo in (("fc", "só pistas"), ("mac", "AC-3"),
                             ("mac_all", "AC-3+AllDiff")):
            cel, comp = [], 0
            for g in lista:
                dom, _, ok = inicializar(g, modo)
                unit = sum(1 for d in dom if POP[d] == 1)
                cel.append(unit)
                comp += int(ok and unit == 81)
            res[rotulo] = dict(celulas=float(np.mean(cel)),
                               pct_resolvido=100 * comp / len(lista))
        todos["propagacao"][nome] = res
        log(f"{nome:14s} " + " | ".join(
            f"{k}: {v['celulas']:.1f} células, {v['pct_resolvido']:.0f}% resolvidos"
            for k, v in res.items()))

    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.7))
    for ax, modo, tit in zip(axs, ("fc", "mac", "mac_all"),
                             ("Pistas removidas dos vizinhos", "Após AC-3",
                              "Após AC-3 + AllDiff")):
        dom, _, _ = inicializar(exemplo, modo)
        desenhar_dominios(ax, dom, tit)
    plt.tight_layout()
    fig.subplots_adjust(top=0.84)
    plt.savefig(f"{SAIDA}/5-fig3_propagacao.png", dpi=170)
    plt.close()

    # ---------- Arquivos de saída ----------
    with open(f"{SAIDA}/resultados-csp.json", "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)
    with open(f"{SAIDA}/resultados-csp.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("RELATÓRIO DE RESULTADOS - PROJETO 5: CSP (SUDOKU)\n")
        f.write("=" * 70 + "\n\n" + "\n".join(linhas) + "\n")
    print(f"\nArquivos salvos com sucesso na pasta ./{SAIDA}:\n"
          "  - 5-fig1_sudoku.png\n"
          "  - 5-fig2_desempenho.png\n"
          "  - 5-fig3_propagacao.png\n"
          "  - resultados-csp.json\n"
          "  - resultados-csp.txt\n")


if __name__ == "__main__":
    main()