# ADR-003: Estrategia de Feature Engineering (Sprint 1)

## Contexto
Para o modelo de ranking de telefones, precisamos de variaveis que descrevam o comportamento historico e as caracteristicas tecnicas de cada contato. O dataset original continha buckets de acionamentos (30d, 60d, 90d), mas nao variaveis diretas de Recencia ou Tipo de Linha.

## Decisao e Motivacao

A IA implementou as seguintes transformacoes no script `feature_engineering.py`:

1. **Recencia (Buckets: 30, 60, 90, 120):**
    - **Decisao:** Criada a partir da verificacao hierarquica dos buckets de acionamento.
    - **Motivacao:** Em recuperacao de credito, o tempo desde o ultimo contato e o preditor mais forte de sucesso. Como nao temos timestamps exatos, a discretizacao em buckets permite que o modelo capture a degradacao da probabilidade de contato ao longo do tempo.

2. **Tipo de Linha (Movel vs Fixo):**
    - **Decisao:** Derivada da posicao do digito '9' no numero.
    - **Motivacao:** Telefones moveis possuem maior taxa de "Alo", mas telefones fixos (comerciais ou residenciais) costumam ter maior taxa de CPC em horarios especificos. Separar essas categorias ajuda o modelo a ajustar o ranking por perfil de linha.

3. **Operadora (Proxy via Prefixo):**
    - **Decisao:** Extracao dos primeiros 3 digitos (moveis) ou 2 (fixos).
    - **Motivacao:** Diferentes operadoras possuem politicas distintas de bloqueio de chamadas (spam) e qualidade de sinal em determinadas regioes. O prefixo atua como uma feature categorica que captura essas nuances operacionais.

4. **Frequencia (Acionamentos Totais):**
    - **Decisao:** Mapeada com capping (percentil 99).
    - **Motivacao:** Indica o nivel de "exaustao" do telefone. Um numero com excesso de acionamentos e nenhum sucesso (CPC) sinaliza um contato improdutivo que deve ser penalizado no ranking final.

5. **Total de Telefones por Cliente:**
    - **Decisao:** Contagem de telefones unicos por `id_pessoa`.
    - **Motivacao:** Clientes com muitos telefones cadastrados podem indicar maior dificuldade de localizacao ou dados desatualizados. Essa feature ajuda o modelo a entender o contexto do cliente, nao apenas do telefone isolado.

## Consequencia
- O dataset agora possui features numericas e categoricas prontas para treinamento.
- O uso de operacoes vetorizadas no Pandas garantiu que o processamento de 1 milhao de linhas fosse realizado em poucos segundos.
- O dataset final foi salvo como `dataset_features.parquet` para preservar os tipos de dados e otimizar o espaco.
