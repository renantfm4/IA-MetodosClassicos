"""PROJETO 2 - BUSCA NÃO INFORMADA (Métodos Clássicos de IA)

=====================================================
Problema: navegação em um labirinto em grade com paredes e terrenos de custos
diferentes (grama = custo 1, lama = custo 5). Um agente sai do canto superior
esquerdo e deve chegar ao canto inferior direito.

Algoritmos (nenhum usa conhecimento sobre a posição do objetivo):
  - BFS  (busca em largura)
  - DFS  (busca em profundidade)
  - DLS  (busca em profundidade limitada)
  - IDDFS (aprofundamento iterativo)
  - UCS  (busca de custo uniforme)
  - BDS  (busca bidirecional em largura)

Como executar:
    python busca_nao_informada.py

Saídas (pasta ./resultados):
  - 2-fig1_labirinto_exploracao.png
  - 2-fig2_comparacao.png
  - 2-fig3_escalabilidade.png
  - resultados-nao-informada.json
  - resultados-nao-informada.txt

Dependências: numpy, matplotlib (o restante é biblioteca padrão do Python).
"""

import heapq
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
random.seed(42)
rng = np.random.default_rng(42)

LIVRE, PAREDE, LAMA = 0, 1, 2
CUSTO_TERRENO = {LIVRE: 1, LAMA: 5}
# Ordem fixa de prioridade dos movimentos (cima, baixo, esquerda, direita).
# Ela influencia DFS e DLS, que dependem da ordem de expansão.
MOVS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# =====================================================================
# PARTE 1 - O PROBLEMA (labirinto em grade)
# =====================================================================
def gerar_labirinto(n=30, p_parede=0.25, p_lama=0.20):
    """Gera grade n x n. Início (0,0) e objetivo (n-1,n-1) ficam livres.
    Regera até existir pelo menos um caminho entre os dois."""
    while True:
        g = np.zeros((n, n), dtype=int)
        g[rng.random((n, n)) < p_parede] = PAREDE
        lama = (rng.random((n, n)) < p_lama) & (g == LIVRE)
        g[lama] = LAMA
        g[0, 0] = g[n - 1, n - 1] = LIVRE
        if bfs(g, (0, 0), (n - 1, n - 1)) is not None:
            return g


def vizinhos(pos, grade):
    """Células vizinhas acessíveis e o custo de ENTRAR em cada uma."""
    n = grade.shape[0]
    r, c = pos
    saida = []
    for dr, dc in MOVS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n and grade[nr, nc] != PAREDE:
            saida.append(((nr, nc), CUSTO_TERRENO[grade[nr, nc]]))
    return saida


def custo_do_caminho(caminho, grade):
    """Soma dos custos de terreno das células visitadas (exceto a inicial)."""
    return sum(CUSTO_TERRENO[grade[r, c]] for r, c in caminho[1:])


def reconstruir(no):
    """Reconstrói o caminho a partir de um nó (estado, profundidade, pai)."""
    caminho = []
    while no is not None:
        caminho.append(no[0])
        no = no[2]
    return caminho[::-1]


def resultado(grade, caminho, expandidos, mem_max, t0, explorados):
    """Empacota as métricas de uma busca."""
    return dict(
        caminho=caminho,
        passos=len(caminho) - 1 if caminho else None,
        custo=custo_do_caminho(caminho, grade) if caminho else None,
        expandidos=expandidos,
        mem_max=mem_max,
        tempo=time.perf_counter() - t0,
        explorados=explorados,
    )


# =====================================================================
# PARTE 2 - ALGORITMOS DE BUSCA NÃO INFORMADA
# =====================================================================
def bfs(grade, ini, obj):
    """Busca em largura: fila FIFO. Ótima em NÚMERO DE PASSOS."""
    t0 = time.perf_counter()
    fila = deque([(ini, 0, None)])
    visitados = {ini}  # marca ao gerar, evita duplicatas na fronteira
    exp, mem, explorados = 0, 1, set()
    while fila:
        no = fila.popleft()
        exp += 1
        explorados.add(no[0])
        if no[0] == obj:
            return resultado(grade, reconstruir(no), exp, mem, t0, explorados)
        for viz, _ in vizinhos(no[0], grade):
            if viz not in visitados:
                visitados.add(viz)
                fila.append((viz, no[1] + 1, no))
        mem = max(mem, len(fila))
    return None


def dfs(grade, ini, obj):
    """Busca em profundidade (versão em grafo): pilha LIFO + conjunto de
    visitados. Pouca memória, mas NÃO é ótima."""
    t0 = time.perf_counter()
    pilha = [(ini, 0, None)]
    visitados = set()
    exp, mem, explorados = 0, 1, set()
    while pilha:
        no = pilha.pop()
        if no[0] in visitados:
            continue
        visitados.add(no[0])
        exp += 1
        explorados.add(no[0])
        if no[0] == obj:
            return resultado(grade, reconstruir(no), exp, mem, t0, explorados)
        # empilha ao contrário para que o 1º movimento de MOVS seja o 1º a sair
        for viz, _ in reversed(vizinhos(no[0], grade)):
            if viz not in visitados:
                pilha.append((viz, no[1] + 1, no))
        mem = max(mem, len(pilha))
    return None


def dls(grade, ini, obj, limite):
    """Busca em profundidade limitada. Um estado é reexpandido somente se
    alcançado por um caminho MAIS RASO (garante que o limite não esconda
    soluções). Retorna (resultado|None, expandidos, mem_max, explorados)."""
    pilha = [(ini, 0, None)]
    melhor_prof = {}
    exp, mem, explorados = 0, 1, set()
    while pilha:
        no = pilha.pop()
        est, prof, _ = no
        if est in melhor_prof and melhor_prof[est] <= prof:
            continue
        melhor_prof[est] = prof
        exp += 1
        explorados.add(est)
        if est == obj:
            return no, exp, mem, explorados
        if prof < limite:
            for viz, _ in reversed(vizinhos(est, grade)):
                pilha.append((viz, prof + 1, no))
        mem = max(mem, len(pilha))
    return None, exp, mem, explorados


def dls_publico(grade, ini, obj, limite):
    """Interface de DLS com as mesmas métricas dos demais algoritmos."""
    t0 = time.perf_counter()
    no, exp, mem, expl = dls(grade, ini, obj, limite)
    caminho = reconstruir(no) if no else None
    return resultado(grade, caminho, exp, mem, t0, expl)


def iddfs(grade, ini, obj):
    """Aprofundamento iterativo: DLS com limite 0, 1, 2, ... Une a pouca
    memória da DFS com a otimalidade (em passos) da BFS."""
    t0 = time.perf_counter()
    exp_total, mem, explorados = 0, 1, set()
    for limite in range(grade.size):
        no, exp, m, expl = dls(grade, ini, obj, limite)
        exp_total += exp
        mem = max(mem, m)
        explorados |= expl
        if no is not None:
            return resultado(grade, reconstruir(no), exp_total, mem, t0, explorados)
    return None


def ucs(grade, ini, obj):
    """Busca de custo uniforme (Dijkstra): expande sempre o nó de menor g(n).
    Ótima em CUSTO TOTAL de terreno."""
    t0 = time.perf_counter()
    cont = itertools.count()
    fronteira = [(0, next(cont), (ini, 0, None))]
    melhor_g = {ini: 0}
    fechados = set()
    mem, explorados = 1, set()
    while fronteira:
        g, _, no = heapq.heappop(fronteira)
        if no[0] in fechados:
            continue  # entrada obsoleta
        fechados.add(no[0])
        explorados.add(no[0])
        if no[0] == obj:
            return resultado(grade, reconstruir(no), len(fechados), mem, t0, explorados)
        for viz, c in vizinhos(no[0], grade):
            ng = g + c
            if viz not in melhor_g or ng < melhor_g[viz]:
                melhor_g[viz] = ng
                heapq.heappush(fronteira, (ng, next(cont), (viz, no[1] + 1, no)))
        mem = max(mem, len(fronteira))
    return None


def bds(grade, ini, obj):
    """Busca bidirecional em largura: duas BFS (do início e do objetivo)
    que se encontram no meio. Expande por níveis completos, sempre pelo lado
    de menor fronteira, e escolhe o melhor encontro do nível."""
    t0 = time.perf_counter()
    if ini == obj:
        return resultado(grade, [ini], 1, 1, t0, {ini})
    dist = [{ini: 0}, {obj: 0}]
    pai = [{ini: None}, {obj: None}]
    fronteira = [[ini], [obj]]
    exp, mem = 0, 2
    while fronteira[0] and fronteira[1]:
        lado = 0 if len(fronteira[0]) <= len(fronteira[1]) else 1
        outro = 1 - lado
        nova, melhor = [], (float("inf"), None)
        for est in fronteira[lado]:
            exp += 1
            for viz, _ in vizinhos(est, grade):  # grafo não direcionado
                if viz not in dist[lado]:
                    dist[lado][viz] = dist[lado][est] + 1
                    pai[lado][viz] = est
                    nova.append(viz)
                if viz in dist[outro]:
                    total = dist[lado][viz] + dist[outro][viz]
                    if total < melhor[0]:
                        melhor = (total, viz)
        fronteira[lado] = nova
        mem = max(mem, len(fronteira[0]) + len(fronteira[1]))
        if melhor[1] is not None:
            meio = melhor[1]
            parte_a, e = [], meio
            while e is not None:
                parte_a.append(e)
                e = pai[0][e]
            parte_b, e = [], pai[1][meio]
            while e is not None:
                parte_b.append(e)
                e = pai[1][e]
            caminho = parte_a[::-1] + parte_b
            return resultado(grade, caminho, exp, mem, t0, set(dist[0]) | set(dist[1]))
    return None


ALGORITMOS = {
    "BFS": bfs,
    "DFS": dfs,
    "IDDFS": iddfs,
    "UCS": ucs,
    "BDS": bds,
}


# =====================================================================
# PARTE 3 - VISUALIZAÇÃO
# =====================================================================
def desenhar_labirinto(ax, grade, res, titulo):
    """Paredes em cinza-escuro, lama em marrom-claro, células expandidas em
    azul e o caminho encontrado em vermelho."""
    n = grade.shape[0]
    img = np.ones((n, n, 3))
    img[grade == PAREDE] = (0.2, 0.2, 0.2)
    img[grade == LAMA] = (0.85, 0.72, 0.5)
    for (r, c) in res["explorados"]:
        base = img[r, c]
        img[r, c] = 0.45 * base + 0.55 * np.array((0.45, 0.65, 0.9))
    ax.imshow(img, interpolation="nearest")
    if res["caminho"]:
        ys, xs = zip(*res["caminho"])
        ax.plot(xs, ys, color="#D62728", lw=1.8)
    ax.plot(0, 0, "go", ms=6)
    ax.plot(n - 1, n - 1, "b*", ms=10)
    ax.set_title(titulo, fontsize=8)
    ax.axis("off")


def resumo(res):
    """Texto curto com passos, custo e nós expandidos de um resultado."""
    if res["caminho"] is None:
        return f"sem solução | exp={res['expandidos']}"
    return f"passos={res['passos']} custo={res['custo']} | exp={res['expandidos']}"


# =====================================================================
# PARTE 4 - EXPERIMENTOS
# =====================================================================
def main():
    """Executa os 3 experimentos, gera as figuras e salva os resultados."""
    todos = {}
    linhas = []

    def log(msg=""):
        print(msg)
        linhas.append(msg)

    # ---------- Experimento 1: um labirinto 30x30 ----------
    lado = 30
    grade = gerar_labirinto(lado)
    ini, obj = (0, 0), (lado - 1, lado - 1)
    res_ex = {nome: f(grade, ini, obj) for nome, f in ALGORITMOS.items()}
    otimo_passos = res_ex["BFS"]["passos"]
    limite_curto = otimo_passos // 2
    res_ex["DLS (limite curto)"] = dls_publico(grade, ini, obj, limite_curto)

    log(f"=== Experimento 1: labirinto {lado}x{lado} (exemplo) ===")
    todos["exemplo"] = {}
    for nome, r in res_ex.items():
        log(f"{nome:20s} {resumo(r)} | mem={r['mem_max']} | t={r['tempo']*1000:.2f} ms")
        todos["exemplo"][nome] = {k: r[k] for k in
                                  ("passos", "custo", "expandidos", "mem_max")}
        todos["exemplo"][nome]["tempo_ms"] = r["tempo"] * 1000

    ordem = ["BFS", "DFS", "IDDFS", "UCS", "BDS", "DLS (limite curto)"]
    fig, axs = plt.subplots(2, 3, figsize=(10, 7))
    for ax, nome in zip(axs.ravel(), ordem):
        r = res_ex[nome]
        t = nome if nome != "DLS (limite curto)" else f"DLS (limite={limite_curto})"
        desenhar_labirinto(ax, grade, r, f"{t}\n{resumo(r)}")
    plt.tight_layout()
    plt.savefig(f"{SAIDA}/2-fig1_labirinto_exploracao.png", dpi=170)
    plt.close()

    # DLS com vários limites no mesmo labirinto
    log(f"\n--- DLS no mesmo labirinto (passos ótimos = {otimo_passos}) ---")
    todos["dls_limites"] = {}
    for lim in (10, 30, 50, otimo_passos - 1, otimo_passos, 100, 200):
        r = dls_publico(grade, ini, obj, lim)
        info = dict(achou=r["caminho"] is not None, passos=r["passos"],
                    custo=r["custo"], expandidos=r["expandidos"])
        todos["dls_limites"][str(lim)] = info
        log(f"limite={lim:4d}  achou={str(info['achou']):5s} passos={info['passos']} "
            f"custo={info['custo']} exp={info['expandidos']}")

    # ---------- Experimento 2: 50 labirintos 30x30 ----------
    log("\n=== Experimento 2: média em 50 labirintos 30x30 ===")
    ac = {nome: dict(exp=[], passos=[], custo=[], mem=[], tempo=[],
                     opt_p=0, opt_c=0) for nome in ALGORITMOS}
    for _ in range(50):
        g = gerar_labirinto(lado)
        rs = {nome: f(g, ini, obj) for nome, f in ALGORITMOS.items()}
        melhor_p = min(r["passos"] for r in rs.values())
        melhor_c = min(r["custo"] for r in rs.values())
        for nome, r in rs.items():
            a = ac[nome]
            a["exp"].append(r["expandidos"])
            a["passos"].append(r["passos"])
            a["custo"].append(r["custo"])
            a["mem"].append(r["mem_max"])
            a["tempo"].append(r["tempo"] * 1000)
            a["opt_p"] += r["passos"] == melhor_p
            a["opt_c"] += r["custo"] == melhor_c
    todos["media_50"] = {}
    log(f"{'Algoritmo':8s}|{'Exp.':>8s}|{'Mem.':>8s}|{'Tempo ms':>9s}|"
        f"{'Passos':>8s}|{'Custo':>8s}|{'%opt passos':>12s}|{'%opt custo':>11s}")
    for nome, a in ac.items():
        d = dict(expandidos=float(np.mean(a["exp"])), mem=float(np.mean(a["mem"])),
                 tempo_ms=float(np.mean(a["tempo"])), passos=float(np.mean(a["passos"])),
                 custo=float(np.mean(a["custo"])),
                 pct_otimo_passos=100 * a["opt_p"] / 50,
                 pct_otimo_custo=100 * a["opt_c"] / 50)
        todos["media_50"][nome] = d
        log(f"{nome:8s}|{d['expandidos']:8.1f}|{d['mem']:8.1f}|{d['tempo_ms']:9.2f}|"
            f"{d['passos']:8.1f}|{d['custo']:8.1f}|{d['pct_otimo_passos']:12.0f}|"
            f"{d['pct_otimo_custo']:11.0f}")

    nomes = list(ALGORITMOS)
    cores = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#72B7B2"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, chave, tit, log_y in (
        (axs[0], "expandidos", "Nós expandidos (média)", True),
        (axs[1], "mem", "Memória: fronteira máxima (média)", True),
        (axs[2], "custo", "Custo de terreno do caminho (média)", False),
    ):
        vals = [todos["media_50"][n][chave] for n in nomes]
        barras = ax.bar(nomes, vals, color=cores)
        ax.set_title(tit, fontsize=10)
        if log_y:
            ax.set_yscale("log")
        for b in barras:
            ax.annotate(f"{b.get_height():.0f}", (b.get_x() + b.get_width() / 2,
                        b.get_height()), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{SAIDA}/2-fig2_comparacao.png", dpi=170)
    plt.close()

    # ---------- Experimento 3: escalabilidade ----------
    log("\n=== Experimento 3: escalabilidade (15 labirintos por tamanho) ===")
    tamanhos = [10, 20, 30, 40, 50]
    esc = {nome: [] for nome in nomes}
    for n in tamanhos:
        acum = {nome: [] for nome in nomes}
        for _ in range(15):
            g = gerar_labirinto(n)
            for nome, f in ALGORITMOS.items():
                acum[nome].append(f(g, (0, 0), (n - 1, n - 1))["expandidos"])
        for nome in nomes:
            esc[nome].append(float(np.mean(acum[nome])))
        log(f"n={n:3d} " + "  ".join(f"{k}={esc[k][-1]:.0f}" for k in nomes))
    todos["escalabilidade"] = dict(tamanhos=tamanhos, expandidos=esc)

    _, ax = plt.subplots(figsize=(5.4, 3.5))
    for nome, cor in zip(nomes, cores):
        ax.plot(tamanhos, esc[nome], "o-", label=nome, color=cor)
    ax.set_yscale("log")
    ax.set_xlabel("Lado da grade (n)")
    ax.set_ylabel("Nós expandidos (média)")
    ax.set_title("Escalabilidade dos algoritmos", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{SAIDA}/2-fig3_escalabilidade.png", dpi=170)
    plt.close()

    # ---------- Arquivos de saída ----------
    with open(f"{SAIDA}/resultados-nao-informada.json", "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)
    with open(f"{SAIDA}/resultados-nao-informada.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("RELATÓRIO DE RESULTADOS - PROJETO 2: BUSCA NÃO INFORMADA\n")
        f.write("=" * 70 + "\n\n" + "\n".join(linhas) + "\n")
    print(f"\nArquivos salvos com sucesso na pasta ./{SAIDA}:\n"
          "  - 2-fig1_labirinto_exploracao.png\n"
          "  - 2-fig2_comparacao.png\n"
          "  - 2-fig3_escalabilidade.png\n"
          "  - resultados-nao-informada.json\n"
          "  - resultados-nao-informada.txt\n")


if __name__ == "__main__":
    main()