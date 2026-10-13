"""PROJETO 3 - BUSCA COMPLEXA (Métodos Clássicos de IA)

=====================================================
Problema: Problema do Caixeiro Viajante (TSP). Dado um conjunto de cidades,
encontrar o menor ciclo que visita todas exatamente uma vez e volta à origem.
O espaço de estados (n-1)!/2 é grande demais para busca sistemática, então
usamos BUSCA LOCAL: algoritmos que mantêm apenas o estado atual (ou poucos
estados) e o melhoram passo a passo, sem guardar o caminho percorrido.

Algoritmos:
  - HC    : Subida de encosta (Hill Climbing, versão "mais íngreme")
  - HC-R  : Subida de encosta com reinícios aleatórios
  - SA    : Têmpera simulada (Simulated Annealing)
  - BEAM  : Busca em feixe local (Local Beam Search)
  - Referências: tour aleatório e heurística do vizinho mais próximo (NN)
  - Ótimo exato (Held-Karp) apenas em instâncias pequenas, para validação

Vizinhança: movimento 2-opt (inverter um trecho do tour).

Como executar:
    python busca_complexa.py          (leva cerca de 1 a 2 minutos)

Saídas (pasta ./resultados):
  - 3-fig1_rotas.png
  - 3-fig2_convergencia.png
  - 3-fig3_distribuicao.png
  - resultados-complexa.json
  - resultados-complexa.txt

Dependências: numpy, matplotlib (o restante é biblioteca padrão do Python).
"""

import itertools
import json
import math
import os
import random
import time

import matplotlib

matplotlib.use("Agg")  # backend sem janela (salva direto em arquivo)
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position

SAIDA = "resultados"
os.makedirs(SAIDA, exist_ok=True)
SEMENTE = 42


# =====================================================================
# PARTE 1 - O PROBLEMA (TSP)
# =====================================================================
def gerar_cidades(n, semente):
    """Gera n cidades aleatórias em um quadrado 100 x 100."""
    return np.random.default_rng(semente).random((n, 2)) * 100


def matriz_distancias(cidades):
    """Matriz n x n de distâncias euclidianas."""
    dif = cidades[:, None, :] - cidades[None, :, :]
    return np.sqrt((dif ** 2).sum(axis=2))


def comprimento(tour, dist):
    """Comprimento do ciclo fechado representado pela permutação 'tour'."""
    t = np.asarray(tour)
    return float(dist[t, np.roll(t, -1)].sum())


def vizinho_mais_proximo(dist, inicio=0):
    """Heurística construtiva gulosa: sempre ir à cidade não visitada mais
    próxima. Usada apenas como referência (não é busca local)."""
    n = len(dist)
    tour = [inicio]
    livres = set(range(n)) - {inicio}
    while livres:
        prox = min(livres, key=lambda c: dist[tour[-1]][c])
        tour.append(prox)
        livres.remove(prox)
    return tour


def held_karp(dist):
    """Solução EXATA por programação dinâmica (O(n^2 * 2^n)). Só viável para
    n pequeno (aqui n = 12); serve para medir o quão perto da solução ótima
    os algoritmos de busca local chegam."""
    n = len(dist)
    d = dist.tolist()
    inf = float("inf")
    # dp[(mascara, j)] = menor custo partindo de 0, visitando 'mascara', fim em j
    dp = {(1 << j, j): d[0][j] for j in range(1, n)}
    for tam in range(2, n):
        for sub in itertools.combinations(range(1, n), tam):
            mascara = sum(1 << j for j in sub)
            for j in sub:
                ant = mascara ^ (1 << j)
                dp[(mascara, j)] = min(
                    dp[(ant, k)] + d[k][j] for k in sub if k != j
                )
    completa = (1 << n) - 2
    return min(dp.get((completa, j), inf) + d[j][0] for j in range(1, n))


# =====================================================================
# PARTE 2 - VIZINHANÇA 2-OPT
# =====================================================================
def deltas_2opt(tour, dist, idx_i, idx_j):
    """Variação de comprimento de TODOS os movimentos 2-opt do tour, de forma
    vetorizada. O movimento (i, j) inverte o trecho tour[i..j] e troca as
    arestas (a,b) e (c,d) por (a,c) e (b,d):
        delta = d(a,c) + d(b,d) - d(a,b) - d(c,d)
    A posição 0 fica fixa (o ciclo não tem início, então não perde nada)."""
    n = len(tour)
    a = tour[idx_i - 1]
    b = tour[idx_i]
    c = tour[idx_j]
    d = tour[(idx_j + 1) % n]
    return dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]


def aplicar_2opt(tour, i, j):
    """Retorna novo tour com o trecho [i..j] invertido."""
    novo = tour.copy()
    novo[i:j + 1] = novo[i:j + 1][::-1]
    return novo


def indices_2opt(n):
    """Pares (i, j), 1 <= i < j <= n-1, de todos os movimentos 2-opt."""
    idx_i, idx_j = np.triu_indices(n - 1, k=1)
    return idx_i + 1, idx_j + 1


# =====================================================================
# PARTE 3 - ALGORITMOS DE BUSCA LOCAL
# =====================================================================
def hill_climbing(dist, rng, tour0=None):
    """Subida de encosta (mais íngreme): a cada passo avalia TODOS os
    vizinhos 2-opt e move para o melhor; para quando nenhum melhora
    (ótimo local). Retorna (tour, comprimento, avaliações, histórico)."""
    n = len(dist)
    tour = rng.permutation(n) if tour0 is None else tour0.copy()
    comp = comprimento(tour, dist)
    idx_i, idx_j = indices_2opt(n)
    avals = 0
    hist = [(0, comp)]
    while True:
        dl = deltas_2opt(tour, dist, idx_i, idx_j)
        avals += len(dl)
        k = int(np.argmin(dl))
        if dl[k] >= -1e-9:
            break  # nenhum vizinho melhora: ótimo local
        tour = aplicar_2opt(tour, idx_i[k], idx_j[k])
        comp += dl[k]
        hist.append((avals, comp))
    return tour, float(comp), avals, hist


def hill_climbing_reinicios(dist, rng, reinicios=20):
    """Subida de encosta com reinícios aleatórios: repete HC a partir de
    tours aleatórios diferentes e guarda o melhor ótimo local encontrado."""
    melhor_tour, melhor, avals_tot = None, float("inf"), 0
    hist = []
    for _ in range(reinicios):
        tour, comp, avals, _ = hill_climbing(dist, rng)
        avals_tot += avals
        if comp < melhor:
            melhor_tour, melhor = tour, comp
        hist.append((avals_tot, melhor))
    return melhor_tour, melhor, avals_tot, hist


def simulated_annealing(dist, semente, alpha=0.995, registrar=False):
    """Têmpera simulada. Escolhe um vizinho 2-opt ALEATÓRIO; aceita se
    melhora, ou com probabilidade exp(-delta/T) se piora. A temperatura
    decai geometricamente (T <- alpha * T). T inicial = distância média entre
    cidades; T final = T0 * 1e-4; n tentativas por temperatura."""
    rnd = random.Random(semente)
    n = len(dist)
    d = dist.tolist()
    tour = list(np.random.default_rng(semente).permutation(n))
    comp = comprimento(tour, dist)
    melhor_tour, melhor = tour.copy(), comp
    temp0 = float(dist.sum() / (n * (n - 1)))
    temp, temp_min = temp0, temp0 * 1e-4
    avals = 0
    hist = []
    while temp > temp_min:
        for _ in range(n):
            i = rnd.randint(1, n - 1)
            j = rnd.randint(1, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            a, b = tour[i - 1], tour[i]
            c, e = tour[j], tour[(j + 1) % n]
            delta = d[a][c] + d[b][e] - d[a][b] - d[c][e]
            avals += 1
            if delta < 0 or rnd.random() < math.exp(-delta / temp):
                tour[i:j + 1] = tour[i:j + 1][::-1]
                comp += delta
                if comp < melhor - 1e-9:
                    melhor, melhor_tour = comp, tour.copy()
        temp *= alpha
        if registrar:
            hist.append((avals, comp, melhor))
    return np.array(melhor_tour), float(melhor), avals, hist


def beam_search(dist, rng, k=10):
    """Busca em feixe local: mantém k estados. A cada passo gera os melhores
    sucessores 2-opt de cada estado, junta tudo e fica com os k melhores
    (distintos). Para quando o conjunto não muda (todos em ótimo local)."""
    n = len(dist)
    idx_i, idx_j = indices_2opt(n)
    estados = []
    for _ in range(k):
        t = rng.permutation(n)
        estados.append((comprimento(t, dist), t))
    avals = 0
    hist = [(0, min(e[0] for e in estados))]
    while True:
        pool = {tuple(t): (c, t) for c, t in estados}  # mantém os atuais
        for comp, tour in estados:
            dl = deltas_2opt(tour, dist, idx_i, idx_j)
            avals += len(dl)
            for m in np.argsort(dl)[:k]:  # k melhores movimentos deste estado
                if dl[m] < -1e-9:
                    novo = aplicar_2opt(tour, idx_i[m], idx_j[m])
                    pool[tuple(novo)] = (comp + dl[m], novo)
        novos = sorted(pool.values(), key=lambda x: x[0])[:k]
        mudou = {tuple(t) for _, t in novos} != {tuple(t) for _, t in estados}
        estados = novos
        hist.append((avals, estados[0][0]))
        if not mudou:
            break
    return estados[0][1], float(estados[0][0]), avals, hist


def executar(nome, dist, semente):
    """Roda um algoritmo (uma execução) e devolve métricas + histórico."""
    rng = np.random.default_rng(semente)
    t0 = time.perf_counter()
    if nome == "HC":
        tour, comp, avals, hist = hill_climbing(dist, rng)
    elif nome == "HC-R":
        tour, comp, avals, hist = hill_climbing_reinicios(dist, rng)
    elif nome == "SA":
        tour, comp, avals, hist = simulated_annealing(dist, semente, registrar=True)
    else:
        tour, comp, avals, hist = beam_search(dist, rng)
    return dict(tour=[int(x) for x in tour], comp=comp, avals=avals,
                tempo=time.perf_counter() - t0, hist=hist)


ALGS = ["HC", "HC-R", "SA", "BEAM"]
NOMES = {"HC": "Hill Climbing", "HC-R": "HC c/ reinícios",
         "SA": "Têmpera simulada", "BEAM": "Feixe local (k=10)"}


# =====================================================================
# PARTE 4 - EXPERIMENTOS, GRÁFICOS E ARQUIVOS DE SAÍDA
# =====================================================================
def desenhar_rota(ax, cidades, tour, titulo):
    """Desenha o ciclo sobre as cidades."""
    t = list(tour) + [tour[0]]
    ax.plot(cidades[t, 0], cidades[t, 1], "-", color="#4C78A8", lw=1.2)
    ax.plot(cidades[:, 0], cidades[:, 1], "o", color="#D62728", ms=3.5)
    ax.plot(cidades[tour[0], 0], cidades[tour[0], 1], "s", color="green", ms=6)
    ax.set_title(titulo, fontsize=8)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    """Executa os experimentos, gera as figuras e salva os resultados."""
    todos = {}
    linhas = []

    def log(msg=""):
        print(msg)
        linhas.append(msg)

    # ---------- Experimento 1: validação com solução exata (n = 12) ----------
    log("=== Experimento 1: 5 instâncias com 12 cidades (ótimo exato) ===")
    n_inst, n_exec = 5, 20
    acerto = {a: 0 for a in ALGS}
    gaps = {a: [] for a in ALGS}
    for inst in range(n_inst):
        dist = matriz_distancias(gerar_cidades(12, 1000 + inst))
        otimo = held_karp(dist)
        for alg in ALGS:
            for ex in range(n_exec):
                r = executar(alg, dist, SEMENTE + 100 * inst + ex)
                gap = max(0.0, 100 * (r["comp"] - otimo) / otimo)
                gaps[alg].append(gap)
                acerto[alg] += gap < 1e-6
    todos["exato_n12"] = {}
    for alg in ALGS:
        d = dict(pct_otimo=100 * acerto[alg] / (n_inst * n_exec),
                 gap_medio=float(np.mean(gaps[alg])),
                 gap_max=float(np.max(gaps[alg])))
        todos["exato_n12"][alg] = d
        log(f"{alg:5s} achou o ótimo em {d['pct_otimo']:5.1f}% das execuções | "
            f"gap médio={d['gap_medio']:.2f}% | gap máx={d['gap_max']:.2f}%")

    # ---------- Experimento 2: instância com 50 cidades ----------
    n_cid, n_exec2 = 50, 30
    cidades = gerar_cidades(n_cid, SEMENTE)
    dist = matriz_distancias(cidades)
    nn = vizinho_mais_proximo(dist)
    comp_nn = comprimento(nn, dist)
    rnd_comp = float(np.mean([comprimento(
        np.random.default_rng(s).permutation(n_cid), dist) for s in range(200)]))
    log(f"\n=== Experimento 2: {n_cid} cidades, {n_exec2} execuções por algoritmo ===")
    log(f"Tour aleatório (média): {rnd_comp:.1f} | Vizinho mais próximo: {comp_nn:.1f}")
    runs = {a: [executar(a, dist, SEMENTE + ex) for ex in range(n_exec2)]
            for a in ALGS}
    melhor_conhecido = min(r["comp"] for a in ALGS for r in runs[a])
    log(f"Melhor comprimento conhecido (todas as execuções): {melhor_conhecido:.1f}")
    todos["n50"] = dict(aleatorio=rnd_comp, vizinho_mais_proximo=comp_nn,
                        melhor_conhecido=melhor_conhecido)
    log(f"{'Algoritmo':6s}|{'Média':>8s}|{'Melhor':>8s}|{'Pior':>8s}|{'Desvio':>7s}|"
        f"{'Gap médio':>10s}|{'Avaliações':>11s}|{'Tempo ms':>9s}")
    for alg in ALGS:
        comps = np.array([r["comp"] for r in runs[alg]])
        d = dict(media=float(comps.mean()), melhor=float(comps.min()),
                 pior=float(comps.max()), desvio=float(comps.std()),
                 gap_medio=float(100 * (comps.mean() - melhor_conhecido) / melhor_conhecido),
                 avaliacoes=float(np.mean([r["avals"] for r in runs[alg]])),
                 tempo_ms=float(np.mean([r["tempo"] for r in runs[alg]]) * 1000))
        todos["n50"][alg] = d
        log(f"{alg:6s}|{d['media']:8.1f}|{d['melhor']:8.1f}|{d['pior']:8.1f}|"
            f"{d['desvio']:7.1f}|{d['gap_medio']:9.1f}%|{d['avaliacoes']:11.0f}|"
            f"{d['tempo_ms']:9.1f}")

    # Figura 1: rotas (execução mediana de cada algoritmo)
    fig, axs = plt.subplots(1, 5, figsize=(14, 3.3))
    desenhar_rota(axs[0], cidades, nn, f"Vizinho mais próximo\ncomprimento = {comp_nn:.1f}")
    for ax, alg in zip(axs[1:], ALGS):
        ordem = sorted(runs[alg], key=lambda r: r["comp"])
        r = ordem[len(ordem) // 2]
        desenhar_rota(ax, cidades, r["tour"], f"{NOMES[alg]} (mediana)\ncomprimento = {r['comp']:.1f}")
    plt.tight_layout()
    plt.savefig(f"{SAIDA}/3-fig1_rotas.png", dpi=170)
    plt.close()

    # Figura 2: convergência (execução 0 de cada algoritmo)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    cores = {"HC": "#999999", "HC-R": "#4C78A8", "SA": "#F58518", "BEAM": "#54A24B"}
    for alg in ALGS:
        h = runs[alg][0]["hist"]
        x = [p[0] for p in h]
        y = [p[2] if alg == "SA" else p[1] for p in h]  # SA: melhor até agora
        ax.plot(x, y, label=NOMES[alg], color=cores[alg], lw=1.6)
    ax.axhline(melhor_conhecido, color="k", ls="--", lw=0.8, label="Melhor conhecido")
    ax.set_xscale("log")
    ax.set_ylim(520, 1300)  # foca na fase final (tours aleatórios ficam acima)
    ax.set_xlabel("Avaliações de vizinhos (escala log)")
    ax.set_ylabel("Melhor comprimento até o momento")
    ax.set_title("Convergência (uma execução de cada)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{SAIDA}/3-fig2_convergencia.png", dpi=170)
    plt.close()

    # Figura 3: distribuição dos resultados em 30 execuções
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.boxplot([[r["comp"] for r in runs[a]] for a in ALGS], widths=0.5)
    ax.set_xticklabels([NOMES[a] for a in ALGS])
    ax.axhline(comp_nn, color="#D62728", ls=":", label="Vizinho mais próximo")
    ax.axhline(melhor_conhecido, color="k", ls="--", lw=0.8, label="Melhor conhecido")
    ax.set_ylabel("Comprimento do tour")
    ax.set_title(f"Distribuição em {n_exec2} execuções ({n_cid} cidades)", fontsize=10)
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{SAIDA}/3-fig3_distribuicao.png", dpi=170)
    plt.close()

    # ---------- Experimento 3: sensibilidade ao resfriamento da SA ----------
    log("\n=== Experimento 3: taxa de resfriamento da SA (10 execuções) ===")
    todos["sa_alpha"] = {}
    for alpha in (0.90, 0.95, 0.99, 0.995, 0.999):
        comps, avs, tempos = [], [], []
        for ex in range(10):
            t0 = time.perf_counter()
            _, c, av, _ = simulated_annealing(dist, SEMENTE + ex, alpha=alpha)
            tempos.append(time.perf_counter() - t0)
            comps.append(c)
            avs.append(av)
        d = dict(media=float(np.mean(comps)), melhor=float(np.min(comps)),
                 desvio=float(np.std(comps)), avaliacoes=float(np.mean(avs)),
                 tempo_ms=float(np.mean(tempos) * 1000))
        todos["sa_alpha"][str(alpha)] = d
        log(f"alpha={alpha:<6} média={d['media']:7.1f} melhor={d['melhor']:7.1f} "
            f"desvio={d['desvio']:5.1f} aval={d['avaliacoes']:8.0f} t={d['tempo_ms']:7.1f} ms")

    # ---------- Arquivos de saída ----------
    with open(f"{SAIDA}/resultados-complexa.json", "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)
    with open(f"{SAIDA}/resultados-complexa.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("RELATÓRIO DE RESULTADOS - PROJETO 3: BUSCA COMPLEXA (TSP)\n")
        f.write("=" * 70 + "\n\n" + "\n".join(linhas) + "\n")
    print(f"\nArquivos salvos com sucesso na pasta ./{SAIDA}:\n"
          "  - 3-fig1_rotas.png\n"
          "  - 3-fig2_convergencia.png\n"
          "  - 3-fig3_distribuicao.png\n"
          "  - resultados-complexa.json\n"
          "  - resultados-complexa.txt\n")


if __name__ == "__main__":
    main()