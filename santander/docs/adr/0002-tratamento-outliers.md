# ADR-002: Estrategia de Tratamento de Outliers (EDA Sprint 1)

## Contexto
Durante a Analise Exploratoria (EDA), foram identificadas inconsistencias graves nos dados que podem enviesar o modelo de Machine Learning:
1. **Idade:** Valores negativos (-74) e valores impossiveis (2025).
2. **Acionamentos:** Presenca de telefones com milhares de acionamentos (max: 4760), sugerindo comportamento mecanico ou erro de processamento.
3. **Telefones Ineficientes:** Grande volume (~40k) de telefones com alto volume de tentativas (>50) e zero CPCs.

## Decisao
A IA sugere as seguintes acoes para a fase de Feature Engineering e Treinamento:
1. **Idade:** Filtrar registros com idade < 18 ou > 100 anos, ou substituir pela mediana da UF/Cidade.
2. **Acionamentos:** Aplicar "Capping" (Winsorization) no percentil 99 ou remover outliers extremos (> 3 desvios padrao) para evitar que o modelo aprenda com comportamentos anomalos.
3. **Telefones por Cliente:** Criar uma feature que contabilize a quantidade de telefones por `id_pessoa`, pois clientes com muitos telefones tendem a ser mais dificeis de localizar.

## Consequencia
- O modelo sera mais robusto a ruidos e erros de imputacao de dados.
- As predicoes de ranking serao baseadas em comportamentos humanos reais de contato, melhorando a precisao do CPC esperado.
- Necessidade de implementar um pipeline de limpeza de dados antes do treinamento.
