"""PROJETO 1 - BUSCA INFORMADA (Métodos Clássicos de IA)

=====================================================
Problema: 8-Puzzle resolvido com A* e Busca Gulosa (Greedy Best-First),
comparando três heurísticas admissíveis:
  h1 = peças fora do lugar | h2 = distância de Manhattan |
  h3 = Manhattan + Conflito Linear.

Como executar:
    python busca_informada.py

Saídas (pasta ./resultados):
  - 1-fig1_8puzzle_comparacao.png
  - 1-fig2_8puzzle_solucao.png
  - 1-fig3_dificuldade.png
  - resultados-informada.json
  - resultados-informada.txt

Dependências: numpy, matplotlib (o restante é biblioteca padrão do Python).
"""

import heapq
import itertools
import json
import os
import random
import time

import matplotlib

matplotlib.use("Agg")  # backend sem janela (salva direto em arquivo)
import matplotlib.pyplot as plt
import numpy as np

SAIDA = "resultados"
os.makedirs(SAIDA, exist_ok=True)
random.seed(42)
np.random.seed(42)


# =====================================================================
# PARTE 1 - ALGORITMO GENÉRICO DE BUSCA INFORMADA
# =====================================================================
def busca_informada(inicio, eh_objetivo, vizinhos, h, modo="astar"):
    """Busca informada genérica baseada em fila de prioridade (heap).

    Parâmetros
    ----------
    inicio       : estado inicial (hashable)
    eh_objetivo  : função estado -> bool
    vizinhos     : função estado -> lista de (estado_filho, custo_do_passo)
    h            : função heurística estado -> estimativa do custo restante
    modo         : "astar"  -> f(n) = g(n) + h(n)
                   "greedy" -> f(n) = h(n)

    Retorna
    -------
    dict com: caminho, custo, expandidos, gerados, memoria_max e tempo (s).
    """
    t0 = time.perf_counter()
    contador = itertools.count()  # desempate estável no heap
    g = {inicio: 0}  # melhor custo conhecido até cada estado
    pai = {inicio: None}  # para reconstruir o caminho
    # item do heap: (f, desempate, estado)
    fronteira = [(h(inicio), next(contador), inicio)]
    fechados = set()
    gerados = 1
    mem_max = 1

    while fronteira:
        _, _, atual = heapq.heappop(fronteira)
        if atual in fechados:
            continue  # entrada obsoleta (já expandido com custo melhor)
        fechados.add(atual)

        if eh_objetivo(atual):
            caminho = []
            n = atual
            while n is not None:
                caminho.append(n)
                n = pai[n]
            caminho.reverse()
            return dict(
                caminho=caminho,
                custo=g[atual],
                expandidos=len(fechados),
                gerados=gerados,
                memoria_max=mem_max,
                tempo=time.perf_counter() - t0,
            )

        for filho, custo in vizinhos(atual):
            novo_g = g[atual] + custo
            # só (re)insere se achou caminho melhor até 'filho'
            if filho not in g or novo_g < g[filho]:
                g[filho] = novo_g
                pai[filho] = atual
                f = novo_g + h(filho) if modo == "astar" else h(filho)
                heapq.heappush(fronteira, (f, next(contador), filho))
                gerados += 1
        mem_max = max(mem_max, len(fronteira))

    return None  # sem solução


# =====================================================================
# PARTE 2 - PROBLEMA: 8-PUZZLE
# =====================================================================
OBJETIVO_8P = (1, 2, 3, 4, 5, 6, 7, 8, 0)  # 0 = espaço vazio
POS_OBJ = {v: (i // 3, i % 3) for i, v in enumerate(OBJETIVO_8P)}


def vizinhos_8p(estado):
    """Gera os estados obtidos movendo o espaço vazio (custo 1 por movimento)."""
    i = estado.index(0)
    r, c = divmod(i, 3)
    saida = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            j = nr * 3 + nc
            lst = list(estado)
            lst[i], lst[j] = lst[j], lst[i]
            saida.append((tuple(lst), 1))
    return saida


# ---- Heurísticas (todas admissíveis) --------------------------------
def h_fora_do_lugar(estado):
    """h1: número de peças (exceto o vazio) fora da posição final."""
    return sum(
        1 for i, v in enumerate(estado) if v != 0 and v != OBJETIVO_8P[i]
    )


def h_manhattan(estado):
    """h2: soma das distâncias de Manhattan de cada peça à sua posição final."""
    total = 0
    for i, v in enumerate(estado):
        if v != 0:
            r, c = divmod(i, 3)
            rg, cg = POS_OBJ[v]
            total += abs(r - rg) + abs(c - cg)
    return total


def h_manhattan_conflito_linear(estado):
    """h3: Manhattan + Conflito Linear.

    Duas peças estão em conflito linear se estão na mesma linha (ou coluna) da
    sua posição final, mas em ordem invertida: uma precisa passar pela outra, o
    que obriga pelo menos 2 movimentos extras. Continua admissível.
    """
    total = h_manhattan(estado)
    # conflitos em linhas
    for r in range(3):
        linha = [
            (estado[r * 3 + c], c)
            for c in range(3)
            if estado[r * 3 + c] != 0 and POS_OBJ[estado[r * 3 + c]][0] == r
        ]
        for (a, ca), (b, cb) in itertools.combinations(linha, 2):
            if (
                POS_OBJ[a][1] > POS_OBJ[b][1]
            ):  # a está à esquerda mas deve ir à direita de b
                total += 2
    # conflitos em colunas
    for c in range(3):
        col = [
            (estado[r * 3 + c], r)
            for r in range(3)
            if estado[r * 3 + c] != 0 and POS_OBJ[estado[r * 3 + c]][1] == c
        ]
        for (a, ra), (b, rb) in itertools.combinations(col, 2):
            if POS_OBJ[a][0] > POS_OBJ[b][0]:
                total += 2
    return total


HEURISTICAS_8P = {
    "Peças fora do lugar": h_fora_do_lugar,
    "Manhattan": h_manhattan,
    "Manhattan + Conf. Linear": h_manhattan_conflito_linear,
}


def tem_solucao(estado):
    """No 8-puzzle, metade das permutações é insolúvel (paridade das inversões)."""
    seq = [v for v in estado if v != 0]
    inv = sum(1 for i in range(8) for j in range(i + 1, 8) if seq[i] > seq[j])
    return inv % 2 == 0


def gerar_instancia_8p(embaralhamentos=60):
    """Gera um estado solúvel fazendo movimentos aleatórios a partir do objetivo."""
    estado = OBJETIVO_8P
    anterior = None
    for _ in range(embaralhamentos):
        opcoes = [s for s, _ in vizinhos_8p(estado) if s != anterior]
        anterior, estado = estado, random.choice(opcoes)
    return estado


def desenhar_tabuleiro(ax, estado, titulo=""):
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal")
    ax.axis("off")
    for i, v in enumerate(estado):
        r, c = divmod(i, 3)
        cor = "#f0f0f0" if v == 0 else "#4C78A8"
        ax.add_patch(
            plt.Rectangle((c + 0.04, 2 - r + 0.04), 0.92, 0.92, color=cor)
        )
        if v != 0:
            ax.text(
                c + 0.5,
                2 - r + 0.5,
                str(v),
                ha="center",
                va="center",
                color="white",
                fontsize=15,
                fontweight="bold",
            )
    ax.set_title(titulo, fontsize=8)


def experimento_8puzzle(n_instancias=100):
    """Compara heurísticas e algoritmos em n instâncias aleatórias solúveis."""
    instancias = []
    while len(instancias) < n_instancias:
        e = gerar_instancia_8p(random.randint(25, 80))
        if e != OBJETIVO_8P and tem_solucao(e):
            instancias.append(e)

    resultados = {}
    bruto = {}  # dados por instância (usados no gráfico por dificuldade)
    for nome, h in HEURISTICAS_8P.items():
        for modo in ("astar", "greedy"):
            exp, tempo, custo, mem = [], [], [], []
            for e in instancias:
                r = busca_informada(
                    e, lambda s: s == OBJETIVO_8P, vizinhos_8p, h, modo
                )
                exp.append(r["expandidos"])
                tempo.append(r["tempo"])
                custo.append(r["custo"])
                mem.append(r["memoria_max"])
            bruto[f"{nome}|{modo}"] = dict(exp=exp, custo=custo)
            resultados[f"{nome}|{modo}"] = dict(
                expandidos_medio=float(np.mean(exp)),
                expandidos_mediana=float(np.median(exp)),
                tempo_medio_ms=float(np.mean(tempo) * 1000),
                custo_medio=float(np.mean(custo)),
                memoria_media=float(np.mean(mem)),
            )
    return instancias, resultados, bruto


# =====================================================================
# PARTE 3 - EXECUÇÃO, GRÁFICOS E ARQUIVOS DE SAÍDA
# =====================================================================
def main():
    todos = {}
    cabecalho = "=== 8-Puzzle (100 instâncias aleatórias) ==="
    print(cabecalho)
    instancias, res8, bruto = experimento_8puzzle(100)
    todos["8puzzle"] = res8

    linhas_saida = [cabecalho]
    for k, v in res8.items():
        linha = (
            f"{k:38s} exp={v['expandidos_medio']:9.1f}  "
            f"t={v['tempo_medio_ms']:8.2f} ms  custo={v['custo_medio']:.2f}"
        )
        print(linha)
        linhas_saida.append(linha)

    # --- Figura 1: nós expandidos (log) e custo, por heurística/algoritmo
    nomes = list(HEURISTICAS_8P.keys())
    x = np.arange(len(nomes))
    w = 0.36
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, chave, tit in (
        (axs[0], "expandidos_medio", "Nós expandidos (média)"),
        (axs[1], "custo_medio", "Custo da solução (nº de movimentos)"),
    ):
        a = [res8[f"{n}|astar"][chave] for n in nomes]
        b = [res8[f"{n}|greedy"][chave] for n in nomes]
        r1 = ax.bar(x - w / 2, a, w, label="A*", color="#4C78A8")
        r2 = ax.bar(x + w / 2, b, w, label="Gulosa", color="#F58518")
        ax.set_xticks(x)
        ax.set_xticklabels(nomes, fontsize=7)
        ax.set_title(tit, fontsize=10)
        ax.legend(fontsize=8)
        if chave == "expandidos_medio":
            ax.set_yscale("log")
        for rect in list(r1) + list(r2):
            ax.annotate(
                f"{rect.get_height():.1f}",
                (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                ha="center",
                va="bottom",
                fontsize=7,
            )
    plt.tight_layout()
    plt.savefig(f"{SAIDA}/1-fig1_8puzzle_comparacao.png", dpi=170)
    plt.close()

    # --- Figura 2: solução passo a passo (instância mais difícil das 30 primeiras)
    exemplo = max(
        instancias[:30],
        key=lambda e: busca_informada(
            e, lambda s: s == OBJETIVO_8P, vizinhos_8p, h_manhattan_conflito_linear
        )["custo"],
    )
    sol = busca_informada(
        exemplo,
        lambda s: s == OBJETIVO_8P,
        vizinhos_8p,
        h_manhattan_conflito_linear,
    )["caminho"]
    passos = np.linspace(0, len(sol) - 1, min(8, len(sol))).astype(int)
    fig, axs = plt.subplots(1, len(passos), figsize=(1.5 * len(passos), 1.9))
    for ax, p in zip(np.atleast_1d(axs), passos):
        desenhar_tabuleiro(ax, sol[p], f"passo {p}")
    plt.tight_layout()
    plt.savefig(f"{SAIDA}/1-fig2_8puzzle_solucao.png", dpi=170)
    plt.close()
    todos["8puzzle_exemplo"] = dict(estado=exemplo, movimentos=len(sol) - 1)

    # --- Figura 3: esforço (A*) em função da profundidade da solução ótima
    prof = np.array(bruto[f"{nomes[2]}|astar"]["custo"])
    faixas = [(0, 12), (13, 18), (19, 24), (25, 40)]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    tabela_faixas = {}
    for nome, cor in zip(nomes, ("#999999", "#4C78A8", "#54A24B")):
        exp = np.array(bruto[f"{nome}|astar"]["exp"])
        med = []
        for lo, hi in faixas:
            msk = (prof >= lo) & (prof <= hi)
            med.append(float(exp[msk].mean()) if msk.any() else np.nan)
        tabela_faixas[nome] = med
        ax.plot(range(len(faixas)), med, "o-", label=nome, color=cor)
    ax.set_xticks(range(len(faixas)))
    ax.set_xticklabels(
        [
            f"{lo}-{hi}\n(n={int(((prof>=lo)&(prof<=hi)).sum())})"
            for lo, hi in faixas
        ],
        fontsize=8,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Profundidade da solução ótima (movimentos)")
    ax.set_ylabel("Nós expandidos pelo A* (média)")
    ax.set_title("Esforço do A* x dificuldade da instância", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{SAIDA}/1-fig3_dificuldade.png", dpi=170)
    plt.close()
    todos["por_dificuldade"] = dict(
        faixas=[f"{lo}-{hi}" for lo, hi in faixas], expandidos=tabela_faixas
    )

    diff_msg = f"\nPor dificuldade: {tabela_faixas}"
    print(diff_msg)
    linhas_saida.append(diff_msg)

    # 1. Salvar JSON solicitado
    with open(
        f"{SAIDA}/resultados-informada.json", "w", encoding="utf-8"
    ) as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)

    # 2. Salvar TXT com a saída formatada do console e dados tabulares
    with open(f"{SAIDA}/resultados-informada.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("RELATÓRIO DE RESULTADOS - PROJETO 1: BUSCA INFORMADA\n")
        f.write("=" * 70 + "\n\n")
        f.write("\n".join(linhas_saida))
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("DETALHAMENTO POR ALGORITMO E HEURÍSTICA:\n")
        f.write("=" * 70 + "\n")
        f.write(
            f"{'Configuração':<35} | {'Exp. Médio':<10} | {'Mediana':<8} | {'Tempo (ms)':<10} | {'Custo':<8} | {'Memória':<8}\n"
        )
        f.write("-" * 90 + "\n")
        for k, v in res8.items():
            f.write(
                f"{k:<35} | {v['expandidos_medio']:<10.1f} | {v['expandidos_mediana']:<8.1f} | {v['tempo_medio_ms']:<10.2f} | {v['custo_medio']:<8.2f} | {v['memoria_media']:<8.1f}\n"
            )
        f.write("-" * 90 + "\n")

    print(
        f"\nArquivos salvos com sucesso na pasta ./{SAIDA}:\n"
        f"  - 1-fig1_8puzzle_comparacao.png\n"
        f"  - 1-fig2_8puzzle_solucao.png\n"
        f"  - 1-fig3_dificuldade.png\n"
        f"  - resultados-informada.json\n"
        f"  - resultados-informada.txt\n"
    )


if __name__ == "__main__":
    main()