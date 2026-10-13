# Métodos Clássicos de Inteligência Artificial

Seis projetos práticos desenvolvidos na disciplina **FGA0221 – Inteligência Artificial** (Engenharia de Software, Universidade de Brasília) para o segundo portifólio da matéria. Cada projeto escolhe um problema concreto, implementa em Python 3 os algoritmos correspondentes, compara-os de forma experimental e gera tabelas, figuras e arquivos de resultados.

**Autor:** Renan Araújo de Souza  
**Professor:** Fabiano Araujo Soares

## Projetos

| # | Projeto | Código | O que faz |
|---|---------|--------|-----------|
| 1 | Busca Informada | `busca_informada.py` | Resolve o 8-Puzzle com A* e Busca Gulosa, comparando três heurísticas (peças fora do lugar, distância de Manhattan e Manhattan com conflito linear) em 100 instâncias. |
| 2 | Busca Não Informada | `busca_nao_informada.py` | Compara BFS, DFS, DLS, IDDFS, UCS e busca bidirecional na navegação de um labirinto 30×30 com terrenos de custos diferentes (grama, lama e paredes), medindo nós expandidos, memória, passos e custo. |
| 3 | Busca Complexa | `busca_complexa.py` | Resolve o Problema do Caixeiro Viajante (TSP) por busca local com vizinhança 2-opt: Hill Climbing, Hill Climbing com reinícios, Têmpera Simulada e Feixe Local, em instâncias de 12 e 50 cidades. |
| 4 | Algoritmo Genético | `algoritmo_genetico.py` | Aplica um algoritmo genético ao Problema da Mochila 0/1 (100 itens), estudando o efeito de seleção, cruzamento, mutação, população, elitismo e reparo, e comparando com o ótimo exato por programação dinâmica. |
| 5 | Satisfação de Restrições (CSP) | `csp_sudoku.py` | Modela o Sudoku como CSP e compara Retrocesso, Verificação Adiante, MRV, MAC (AC-3), MAC com AllDiff e Mínimo de Conflitos em 40 puzzles de quatro níveis de dificuldade. |
| 6 | Base de Conhecimento | `base_conhecimento.py` | Agente para o Mundo de Wumpus 4×4 que raciocina com uma base de conhecimento em lógica proposicional (FNC); compara os motores de inferência (tabela-verdade, resolução e DPLL) e avalia as políticas cautelosa, arriscada e aleatória em 300 mundos. |

## Estrutura do repositório

```
.
├── README.md
├── busca_informada.py
├── busca_nao_informada.py
├── busca_complexa.py
├── algoritmo_genetico.py
├── csp_sudoku.py
├── base_conhecimento.py
└── resultados/
    ├── 1-fig1_8puzzle_comparacao.png
    ├── 1-fig2_8puzzle_solucao.png
    ├── 1-fig3_dificuldade.png
    ├── 2-fig1_labirinto_exploracao.png
    ├── 2-fig2_comparacao.png
    ├── 2-fig3_escalabilidade.png
    ├── 3-fig1_rotas.png
    ├── 3-fig2_convergencia.png
    ├── 3-fig3_distribuicao.png
    ├── 4-fig1_convergencia.png
    ├── 4-fig2_parametros.png
    ├── 4-fig3_solucao.png
    ├── 5-fig1_sudoku.png
    ├── 5-fig2_desempenho.png
    ├── 5-fig3_propagacao.png
    ├── 6-fig1_mundo.png
    ├── 6-fig2_inferencia.png
    ├── 6-fig3_agentes.png
    ├── resultados-informada.json / .txt
    ├── resultados-nao-informada.json / .txt
    ├── resultados-complexa.json / .txt
    ├── resultados-genetico.json / .txt
    ├── resultados-csp.json / .txt
    └── resultados-conhecimento.json / .txt
```

A pasta `resultados/` já contém as saídas geradas pela última execução de cada projeto. Os nomes das figuras começam com o número do projeto (`1-` a `6-`).

## Requisitos

- Python 3.8 ou superior
- [numpy](https://numpy.org/) e [matplotlib](https://matplotlib.org/)

Todo o resto (`heapq`, `collections`, `itertools`, `json`, `random`, `time`...) faz parte da biblioteca padrão do Python.

## Instalação

```powershell
# (opcional) criar e ativar um ambiente virtual
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # Linux / macOS

pip install numpy matplotlib
```

## Como executar

Cada projeto é um script independente. Rode-o a partir da pasta raiz do repositório:

```powershell
python busca_informada.py
python busca_nao_informada.py
python busca_complexa.py
python algoritmo_genetico.py
python csp_sudoku.py
python base_conhecimento.py
```

Cada execução imprime um resumo no terminal e grava, na pasta `resultados/` (criada automaticamente), as figuras `.png` do projeto e os arquivos `resultados-*.json` e `resultados-*.txt` com os números. O projeto 6, por exemplo, leva cerca de 1 a 3 minutos.

| Projeto | Arquivos gerados em `resultados/` |
|---------|-----------------------------------|
| 1 | `1-fig*.png`, `resultados-informada.json/.txt` |
| 2 | `2-fig*.png`, `resultados-nao-informada.json/.txt` |
| 3 | `3-fig*.png`, `resultados-complexa.json/.txt` |
| 4 | `4-fig*.png`, `resultados-genetico.json/.txt` |
| 5 | `5-fig*.png`, `resultados-csp.json/.txt` |
| 6 | `6-fig*.png`, `resultados-conhecimento.json/.txt` |

## Reprodutibilidade

Os códigos usam sementes aleatórias fixas, então as instâncias geradas e os valores relatados podem ser reproduzidos. Os tempos de execução, porém, dependem da máquina e da versão do Python.

## Relatório

Os problemas, os algoritmos, as tabelas e a análise dos resultados de cada projeto estão descritos no relatório da disciplina.