# -*- coding: utf-8 -*-
"""PROJETO 4 - ALGORITMO GENÉTICO (Métodos Clássicos de IA)

=====================================================
Problema: Mochila 0/1 (Knapsack). Dados n itens, cada um com peso e valor,
escolher o subconjunto de maior valor total cujo peso não ultrapasse a
capacidade da mochila. É NP-difícil; aqui há também um algoritmo exato
(programação dinâmica) que fornece o ótimo para validar o genético.

Instância: itens "fortemente correlacionados" (valor = peso + 10), uma
classe conhecida por ser difícil para heurísticas simples.

Algoritmo genético (AG):
  - Cromossomo: vetor binário de n posições (1 = item na mochila)
  - Seleção: torneio (k = 3) ou roleta
  - Cruzamento: 1 ponto, 2 pontos ou uniforme
  - Mutação: inversão de bit (probabilidade pm por gene)
  - Elitismo: os melhores indivíduos passam direto para a próxima geração
  - Restrição de capacidade: REPARO (descarta itens até caber) ou PENALIDADE

Como executar:
    python algoritmo_genetico.py          (leva cerca de 2 a 4 minutos)

Saídas (pasta ./resultados):
  - 4-fig1_convergencia.png
  - 4-fig2_parametros.png
  - 4-fig3_solucao.png
  - resultados-genetico.json
  - resultados-genetico.txt

Dependências: numpy, matplotlib (o restante é biblioteca padrão do Python).
"""

import json
import os
import time

import matplotlib

matplotlib.use("Agg")  # backend sem janela (salva direto em arquivo)
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position

SAIDA = "resultados"
os.makedirs(SAIDA, exist_ok=True)
SEMENTE = 42

# Configuração padrão do AG (cada experimento altera um parâmetro por vez)
CONFIG_AJUSTADA = dict(pop=200, geracoes=500, restricao="penalidade")
CONFIG_PADRAO = dict(
    pop=100, geracoes=200, pc=0.9, pm_rel=1.0,  # pm = pm_rel / n
    selecao="torneio", k_torneio=3, cruzamento="2pontos",
    elite=2, restricao="reparo",
)


# =====================================================================
# PARTE 1 - O PROBLEMA E O ALGORITMO EXATO
# =====================================================================
def gerar_instancia(n, semente):
    """Itens fortemente correlacionados: valor = peso + 10. A capacidade é
    40% do peso total, de modo que cerca de 40% dos itens cabem."""
    rng = np.random.default_rng(semente)
    pesos = rng.integers(5, 51, n)
    valores = pesos + 10
    capacidade = int(0.4 * pesos.sum())
    return pesos, valores, capacidade


def mochila_exata(pesos, valores, cap):
    """Programação dinâmica O(n * capacidade). Retorna (ótimo, itens)."""
    n = len(pesos)
    tab = np.zeros((n + 1, cap + 1), dtype=np.int64)
    for i in range(n):
        tab[i + 1] = tab[i]
        w = pesos[i]
        tab[i + 1, w:] = np.maximum(tab[i, w:], tab[i, :cap + 1 - w] + valores[i])
    escolhidos = np.zeros(n, dtype=np.uint8)
    j = cap
    for i in range(n, 0, -1):
        if tab[i, j] != tab[i - 1, j]:
            escolhidos[i - 1] = 1
            j -= pesos[i - 1]
    return int(tab[n, cap]), escolhidos


def guloso_razao(pesos, valores, cap):
    """Heurística gulosa: itens em ordem decrescente de valor/peso."""
    total, peso = 0, 0
    for i in np.argsort(-valores / pesos):
        if peso + pesos[i] <= cap:
            peso += pesos[i]
            total += valores[i]
    return int(total)


# =====================================================================
# PARTE 2 - COMPONENTES DO ALGORITMO GENÉTICO (vetorizados com numpy)
# =====================================================================
def reparar(pop, pesos, cap, rng):
    """Reparo: nos indivíduos que estouram a capacidade, os itens escolhidos
    são reconsiderados em ordem aleatória; cada item é mantido se ainda
    couber e descartado caso contrário. Indivíduos viáveis não mudam."""
    inviavel = pop @ pesos > cap
    if not inviavel.any():
        return pop
    sub = pop[inviavel]
    ordem = np.argsort(rng.random(sub.shape), axis=1)
    sel = np.take_along_axis(sub, ordem, axis=1)
    pes = pesos[ordem]
    atual = np.zeros(len(sub), dtype=np.int64)
    manter = np.zeros_like(sel)
    for j in range(sel.shape[1]):  # vetorizado sobre os indivíduos
        cabe = (sel[:, j] == 1) & (atual + pes[:, j] <= cap)
        manter[:, j] = cabe
        atual += np.where(cabe, pes[:, j], 0)
    novo = np.zeros_like(sub)
    np.put_along_axis(novo, ordem, manter, axis=1)
    pop = pop.copy()
    pop[inviavel] = novo
    return pop


def avaliar(pop, pesos, valores, cap, restricao):
    """Retorna (aptidão, valor, viável). No reparo, a aptidão é o valor.
    Na penalidade, cada unidade de peso excedente desconta o maior valor
    por unidade de peso da instância."""
    peso = pop @ pesos
    valor = pop @ valores
    viavel = peso <= cap
    if restricao == "penalidade":
        lam = float((valores / pesos).max())
        apt = valor - lam * np.maximum(0, peso - cap)
    else:
        apt = valor.astype(float)
    return apt, valor, viavel


def selecionar(apt, metodo, k, rng):
    """Retorna os índices dos pais selecionados (um por indivíduo)."""
    tam = len(apt)
    if metodo == "roleta":
        p = apt - apt.min() + 1e-9  # desloca para ficar positivo
        return rng.choice(tam, size=tam, p=p / p.sum())
    cand = rng.integers(0, tam, (tam, k))  # torneio de tamanho k
    return cand[np.arange(tam), np.argmax(apt[cand], axis=1)]


def cruzar(pais_a, pais_b, tipo, pc, rng):
    """Cruzamento vetorizado de pares de pais; com probabilidade 1-pc os
    filhos são cópias dos pais."""
    m, n = pais_a.shape
    pos = np.arange(n)
    if tipo == "1ponto":
        c1 = rng.integers(1, n, m)
        mascara = pos[None, :] < c1[:, None]
    elif tipo == "2pontos":
        c1 = rng.integers(1, n, m)
        c2 = rng.integers(1, n, m)
        lo, hi = np.minimum(c1, c2), np.maximum(c1, c2)
        mascara = (pos[None, :] >= lo[:, None]) & (pos[None, :] < hi[:, None])
    else:  # uniforme
        mascara = rng.random((m, n)) < 0.5
    mascara = mascara & (rng.random(m) < pc)[:, None]
    filho_a = np.where(mascara, pais_b, pais_a)
    filho_b = np.where(mascara, pais_a, pais_b)
    return filho_a, filho_b


def algoritmo_genetico(pesos, valores, cap, semente, **cfg_extra):
    """Executa uma rodada do AG. Retorna dict com a melhor solução VIÁVEL
    encontrada e o histórico por geração."""
    cfg = {**CONFIG_PADRAO, **cfg_extra}
    rng = np.random.default_rng(semente)
    n = len(pesos)
    tam, pm = cfg["pop"], cfg["pm_rel"] / n
    dens = cap / pesos.sum()  # densidade inicial ~ fração que cabe
    pop = (rng.random((tam, n)) < dens).astype(np.uint8)
    if cfg["restricao"] == "reparo":
        pop = reparar(pop, pesos, cap, rng)

    melhor_val, melhor_ind = 0, np.zeros(n, dtype=np.uint8)
    hist_melhor, hist_media = [], []
    t0 = time.perf_counter()
    for _ in range(cfg["geracoes"]):
        apt, valor, viavel = avaliar(pop, pesos, valores, cap, cfg["restricao"])
        if viavel.any():
            i = int(np.argmax(np.where(viavel, valor, -1)))
            if valor[i] > melhor_val:
                melhor_val, melhor_ind = int(valor[i]), pop[i].copy()
        hist_melhor.append(melhor_val)
        hist_media.append(float(apt.mean()))

        elite = pop[np.argsort(-apt)[:cfg["elite"]]] if cfg["elite"] else None
        pais = selecionar(apt, cfg["selecao"], cfg["k_torneio"], rng)
        a, b = cruzar(pop[pais[0::2]], pop[pais[1::2]], cfg["cruzamento"], cfg["pc"], rng)
        filhos = np.vstack([a, b])[:tam]
        filhos = filhos ^ (rng.random(filhos.shape) < pm).astype(np.uint8)
        if cfg["restricao"] == "reparo":
            filhos = reparar(filhos, pesos, cap, rng)
        if elite is not None:
            filhos[:cfg["elite"]] = elite
        pop = filhos
    return dict(valor=melhor_val, ind=melhor_ind, hist=hist_melhor,
                media=hist_media, tempo=time.perf_counter() - t0,
                avaliacoes=cfg["pop"] * cfg["geracoes"])


def busca_aleatoria(pesos, valores, cap, semente, n_amostras):
    """Referência: gera n_amostras soluções aleatórias viáveis (com o mesmo
    reparo) e guarda a melhor. Usa o mesmo número de avaliações do AG."""
    rng = np.random.default_rng(semente)
    n = len(pesos)
    melhor = 0
    for _ in range(n_amostras // 1000):
        amostra = (rng.random((1000, n)) < cap / pesos.sum()).astype(np.uint8)
        amostra = reparar(amostra, pesos, cap, rng)
        melhor = max(melhor, int((amostra @ valores).max()))
    return melhor


# =====================================================================
# PARTE 3 - EXPERIMENTOS, GRÁFICOS E ARQUIVOS DE SAÍDA
# =====================================================================
def rodar(pesos, valores, cap, otimo, n_exec, **cfg):
    """Executa n_exec rodadas do AG e resume os resultados em % do ótimo."""
    res = [algoritmo_genetico(pesos, valores, cap, SEMENTE + s, **cfg)
           for s in range(n_exec)]
    pct = np.array([100 * r["valor"] / otimo for r in res])
    return dict(media=float(pct.mean()), desvio=float(pct.std()),
                melhor=float(pct.max()), pior=float(pct.min()),
                pct_otimo=float(100 * np.mean(pct >= 100 - 1e-9)),
                tempo_ms=float(np.mean([r["tempo"] for r in res]) * 1000)), res


def main():
    """Executa os experimentos, gera as figuras e salva os resultados."""
    todos = {}
    linhas = []

    def log(msg=""):
        print(msg)
        linhas.append(msg)

    n = 100
    pesos, valores, cap = gerar_instancia(n, SEMENTE)
    otimo, itens_otimo = mochila_exata(pesos, valores, cap)
    guloso = guloso_razao(pesos, valores, cap)
    aval = CONFIG_PADRAO["pop"] * CONFIG_PADRAO["geracoes"]
    aleat = [busca_aleatoria(pesos, valores, cap, SEMENTE + s, aval) for s in range(30)]
    aleat_5x = [busca_aleatoria(pesos, valores, cap, SEMENTE + s, 5 * aval)
                for s in range(30)]
    log(f"=== Instância: {n} itens, capacidade {cap} (peso total {pesos.sum()}) ===")
    log(f"Ótimo exato (PD): {otimo} | Guloso por razão: {guloso} "
        f"({100*guloso/otimo:.2f}%) | Busca aleatória (média de 30): "
        f"{np.mean(aleat):.1f} ({100*np.mean(aleat)/otimo:.2f}%)")
    todos["instancia"] = dict(n=n, capacidade=cap, otimo=otimo, guloso=guloso,
                              aleatoria_media=float(np.mean(aleat)),
                              aleatoria_100k_media=float(np.mean(aleat_5x)))

    # ---------- Experimento 1: AG padrão, 30 execuções ----------
    log("\n=== Experimento 1: AG padrão (30 execuções) ===")
    base, runs = rodar(pesos, valores, cap, otimo, 30)
    todos["baseline"] = base
    log(f"% do ótimo: média={base['media']:.2f} desvio={base['desvio']:.2f} "
        f"melhor={base['melhor']:.2f} pior={base['pior']:.2f} | "
        f"achou o ótimo em {base['pct_otimo']:.0f}% das execuções | "
        f"{base['tempo_ms']:.0f} ms/execução")

    ajust, runs_aj = rodar(pesos, valores, cap, otimo, 30, **CONFIG_AJUSTADA)
    ajust["avaliacoes"] = CONFIG_AJUSTADA["pop"] * CONFIG_AJUSTADA["geracoes"]
    todos["ajustado"] = ajust
    log(f"AG ajustado {CONFIG_AJUSTADA}: média={ajust['media']:.2f} "
        f"desvio={ajust['desvio']:.2f} melhor={ajust['melhor']:.2f} "
        f"pior={ajust['pior']:.2f} | ótimo em {ajust['pct_otimo']:.0f}% | "
        f"{ajust['tempo_ms']:.0f} ms/execução")
    hist_aj = np.array([r["hist"] for r in runs_aj]) / otimo * 100

    hist = np.array([r["hist"] for r in runs]) / otimo * 100
    media = np.array([r["media"] for r in runs])
    gens = np.arange(1, hist.shape[1] + 1)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    axs[0].plot(gens, hist.mean(0), color="#4C78A8", label="AG: melhor até a geração")
    axs[0].fill_between(gens, hist.min(0), hist.max(0), color="#4C78A8", alpha=0.2,
                        label="mín–máx (30 execuções)")
    axs[0].plot(np.arange(1, hist_aj.shape[1] + 1), hist_aj.mean(0), color="#B279A2",
                label="AG ajustado (pop 200, penalidade)")
    axs[0].axhline(100, color="k", ls="--", lw=0.8, label="Ótimo (PD)")
    axs[0].axhline(100 * guloso / otimo, color="#54A24B", ls=":", label="Guloso")
    axs[0].axhline(100 * np.mean(aleat) / otimo, color="#D62728", ls=":",
                   label="Busca aleatória")
    axs[0].set_ylim(92, 100.6)
    axs[0].set_xscale("log")
    axs[0].set_xlabel("Geração (escala log)")
    axs[0].set_ylabel("% do valor ótimo")
    axs[0].set_title("Convergência do melhor indivíduo", fontsize=10)
    axs[0].legend(fontsize=7, loc="lower right")
    axs[1].plot(gens, media.mean(0) / otimo * 100, color="#F58518")
    axs[1].set_xlabel("Geração")
    axs[1].set_ylabel("% do valor ótimo")
    axs[1].set_title("Aptidão média da população", fontsize=10)
    axs[1].grid(alpha=0.3)
    axs[0].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{SAIDA}/4-fig1_convergencia.png", dpi=170)
    plt.close()

    # Figura 3: itens escolhidos pelo AG x ótimo
    melhor_run = max(runs, key=lambda r: r["valor"])
    ind = melhor_run["ind"]
    pior_run = min(runs, key=lambda r: r["valor"])
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, r, titulo in ((axs[0], melhor_run, "Melhor execução"),
                          (axs[1], pior_run, "Pior execução")):
        ia = r["ind"]
        cats = [((ia == 1) & (itens_otimo == 1), "#54A24B", "AG e ótimo"),
                ((ia == 1) & (itens_otimo == 0), "#D62728", "só AG"),
                ((ia == 0) & (itens_otimo == 1), "#F58518", "só ótimo"),
                ((ia == 0) & (itens_otimo == 0), "#BBBBBB", "nenhum")]
        for mask, cor, nome in cats:
            ax.scatter(pesos[mask], valores[mask], c=cor, s=18, label=f"{nome} ({mask.sum()})")
        ax.set_xlabel("Peso do item")
        ax.set_title(f"{titulo}: valor {r['valor']} de {otimo}", fontsize=10)
        ax.legend(fontsize=7)
    axs[0].set_ylabel("Valor do item")
    plt.tight_layout()
    plt.savefig(f"{SAIDA}/4-fig3_solucao.png", dpi=170)
    plt.close()
    todos["solucao"] = dict(melhor=melhor_run["valor"], pior=pior_run["valor"],
                            itens_ag=int(ind.sum()), itens_otimo=int(itens_otimo.sum()))

    # ---------- Experimento 2: estudo de parâmetros ----------
    log("\n=== Experimento 2: efeito dos parâmetros (20 execuções cada) ===")
    grupos = {
        "Seleção": [("torneio (k=3)", {}), ("roleta", dict(selecao="roleta"))],
        "Cruzamento": [("1 ponto", dict(cruzamento="1ponto")),
                       ("2 pontos", {}), ("uniforme", dict(cruzamento="uniforme"))],
        "Mutação (pm × n)": [("0", dict(pm_rel=0.0)), ("0,2", dict(pm_rel=0.2)),
                             ("1", {}), ("5", dict(pm_rel=5.0)),
                             ("20", dict(pm_rel=20.0))],
        "População": [("20", dict(pop=20)), ("50", dict(pop=50)),
                      ("100", {}), ("200", dict(pop=200))],
        "Elitismo / Restrição": [("elitismo 2", {}), ("sem elitismo", dict(elite=0)),
                                 ("reparo", {}),
                                 ("penalidade", dict(restricao="penalidade"))],
    }
    todos["parametros"] = {}
    for grupo, itens in grupos.items():
        todos["parametros"][grupo] = {}
        log(f"-- {grupo} --")
        for nome, cfg in itens:
            r, _ = rodar(pesos, valores, cap, otimo, 20, **cfg)
            todos["parametros"][grupo][nome] = r
            log(f"{nome:14s} média={r['media']:6.2f}% desvio={r['desvio']:.2f} "
                f"melhor={r['melhor']:.2f}% ótimo em {r['pct_otimo']:3.0f}% | "
                f"{r['tempo_ms']:.0f} ms")

    fig, axs = plt.subplots(1, 5, figsize=(14, 3.4), sharey=True)
    for ax, (grupo, dados) in zip(axs, todos["parametros"].items()):
        nomes = list(dados)
        medias = [dados[k]["media"] for k in nomes]
        desv = [dados[k]["desvio"] for k in nomes]
        ax.bar(range(len(nomes)), medias, yerr=desv, capsize=3, color="#4C78A8")
        ax.set_xticks(range(len(nomes)))
        ax.set_xticklabels(nomes, rotation=30, ha="right", fontsize=7)
        ax.set_title(grupo, fontsize=9)
        ax.set_ylim(90, 100.5)
        for i, m in enumerate(medias):
            ax.text(i, 90.3, f"{m:.1f}", ha="center", color="white", fontsize=7)
    axs[0].set_ylabel("% do ótimo (média ± desvio)")
    plt.tight_layout()
    plt.savefig(f"{SAIDA}/4-fig2_parametros.png", dpi=170)
    plt.close()

    # ---------- Experimento 3: escalabilidade ----------
    log("\n=== Experimento 3: tamanho da instância (10 execuções) ===")
    todos["escala"] = {}
    for tam_n in (50, 100, 200, 500):
        p2, v2, c2 = gerar_instancia(tam_n, SEMENTE)
        ot2, _ = mochila_exata(p2, v2, c2)
        gu2 = guloso_razao(p2, v2, c2)
        r, _ = rodar(p2, v2, c2, ot2, 10)
        r["guloso_pct"] = 100 * gu2 / ot2
        todos["escala"][str(tam_n)] = r
        log(f"n={tam_n:4d} AG média={r['media']:.2f}% melhor={r['melhor']:.2f}% "
            f"ótimo em {r['pct_otimo']:3.0f}% | guloso={r['guloso_pct']:.2f}% | "
            f"{r['tempo_ms']:.0f} ms")

    # ---------- Arquivos de saída ----------
    with open(f"{SAIDA}/resultados-genetico.json", "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)
    with open(f"{SAIDA}/resultados-genetico.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("RELATÓRIO DE RESULTADOS - PROJETO 4: ALGORITMO GENÉTICO\n")
        f.write("=" * 70 + "\n\n" + "\n".join(linhas) + "\n")
    print(f"\nArquivos salvos com sucesso na pasta ./{SAIDA}:\n"
          "  - 4-fig1_convergencia.png\n"
          "  - 4-fig2_parametros.png\n"
          "  - 4-fig3_solucao.png\n"
          "  - resultados-genetico.json\n"
          "  - resultados-genetico.txt\n")


if __name__ == "__main__":
    main()