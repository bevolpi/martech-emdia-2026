# Relatorio de Analise Exploratoria (EDA) - Sprint 1

## 1. Estatisticas Gerais
|       |     idade |   qnt_acionamentos_total |   qnt_cpc_total |   qnt_alo_total |
|:------|----------:|-------------------------:|----------------:|----------------:|
| count |    1e+06  |                   1e+06  |         1e+06   |         1e+06   |
| mean  |   55.5735 |                  20.8584 |         2.67904 |         9.67322 |
| std   |   33.3441 |                  44.1642 |         9.8211  |        26.3015  |
| min   |  -74      |                   1      |         0       |         0       |
| 25%   |   35      |                   4      |         0       |         0       |
| 50%   |   45      |                   8      |         0       |         3       |
| 75%   |   60      |                  20      |         0       |         9       |
| max   | 2025      |                4760      |       960       |      2002       |


## 2. Distribuicao de Telefones por Cliente
Quantidade de telefones por cliente:
|   telefone |   count |
|-----------:|--------:|
|          1 |  386689 |
|          2 |  130126 |
|          3 |   50670 |
|          4 |   20333 |
|          5 |    9219 |
|          6 |    4421 |
|          7 |    2288 |
|          8 |    1252 |
|          9 |     724 |
|         10 |     475 |
|         11 |     265 |
|         12 |     186 |
|         13 |     124 |
|         14 |      62 |
|         15 |      50 |
|         16 |      28 |
|         17 |      17 |
|         18 |      12 |
|         19 |       9 |
|         20 |       7 |
|         21 |       1 |
|         22 |       4 |
|         23 |       1 |
|         25 |       1 |


## 3. Analise de CPC
- **Taxa de CPC Global:** 12.84%


## 4. Identificacao de Outliers
- **Limite superior para acionamentos (IQR):** 44.0
- **Quantidade de registros acima do limite:** 104094 (10.41%)
- **Clientes com mais de 5 telefones:** 9927
- **Telefones ineficientes (>50 acionamentos e 0 CPC):** 39760