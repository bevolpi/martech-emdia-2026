# ADR-007: Schema de Exportacao do Output (Padrao Emdia)

## Contexto
O sistema de scoring precisa entregar um arquivo padronizado para a Emdia processar
na camada de discagem. O schema define colunas, tipos e naming do arquivo de forma
que qualquer consumidor downstream possa processar sem transformacoes adicionais.

## Schema Acordado

### Colunas Obrigatorias

| Coluna             | Tipo       | Descricao                                          |
|--------------------|------------|----------------------------------------------------|
| `id_pessoa`        | string     | Identificador unico do cliente                     |
| `telefone`         | string     | Numero de telefone (com DDD, sem formatacao)       |
| `rank_telefone`    | int8       | Posicao no ranking do cliente (1 = melhor score)   |
| `score_conversao`  | float32    | Probabilidade de conversao prevista pelo modelo (0-1) |
| `flag_conversao`   | int8       | Classe predita pelo modelo (0 = nao converte, 1 = converte) |
| `sexo`             | category   | Sexo do cliente (M / F / Desconhecido)             |
| `uf`               | category   | Unidade Federativa do cliente                      |
| `tipo_linha`       | category   | Tipo de linha (Movel / Fixo)                       |
| `idade`            | Int16      | Idade do cliente (inteiro nullable)                |
| `dt_processamento` | string     | Data de processamento no formato YYYY-MM-DD        |

### Regras de Negocio
1. Cada arquivo contem apenas os **Top 5 telefones** de cada cliente, ordenados por `rank_telefone`.
2. `rank_telefone` varia de 1 a 5 dentro de cada `id_pessoa`.
3. Nao pode haver duplicata de `(id_pessoa, telefone)` no mesmo arquivo.
4. `score_conversao` deve estar sempre no intervalo [0.0, 1.0].

### Naming dos Arquivos
- **Parquet (producao):** `output_scored_YYYYMMDD.parquet`
- **CSV (auditoria/compartilhamento):** `output_scored_YYYYMMDD.csv` (separador `;`, encoding `utf-8-sig`)

## Decisao
O script `export_output.py` encapsula toda a logica de transformacao e validacao.
Ele deve ser chamado como ultima etapa do `pipeline_diario.py` apos a escoragem.

## Consequencia
- Qualquer alteracao no schema deve ser refletida neste ADR e no script `export_output.py`.
- O arquivo Parquet e o formato de producao principal.
- O arquivo CSV e gerado para auditoria e compartilhamento sem ferramentas especializadas.
