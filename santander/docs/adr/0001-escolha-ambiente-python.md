# ADR-001: Escolha do Ambiente de Execuo (Python 3 via `py`)

## Contexto
O ambiente local possui o comando `python` configurado para a verso 2.7, que no suporta f-strings e bibliotecas modernas de cincia de dados como `pyarrow` e `streamlit`, alm de apresentar problemas com encoding UTF-8. O projeto foi iniciado com scripts que utilizam f-strings, indicando a necessidade de Python 3.

## Deciso
Utilizaremos o launcher `py` (Python Launcher for Windows) para garantir a execuo com a verso mais recente do Python 3 (3.14.3) instalada no sistema.

## Consequncia
- Todos os comandos de execuo de scripts devem usar `py script.py` em vez de `python script.py`.
- Suporte total a f-strings e bibliotecas modernas.
- Codificao UTF-8 nativa, evitando erros de sintaxe em comentrios e strings com caracteres especiais.
