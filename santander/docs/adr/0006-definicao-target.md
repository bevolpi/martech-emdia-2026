# ADR-006: Definicao da Variavel-Alvo (Target Label)

## Contexto
O objetivo do sistema e maximizar tanto o contato efetivo (CPC) quanto a conversao final (promessa de pagamento). Anteriormente, estavamos utilizando apenas o CPC como alvo, mas a regra de negocio exige a consideracao de ambos.

## Decisao e Motivacao

1. **Variavel Combinada (`target_label`):**
    - **Decisao:** Criamos a variavel `target_label` definida como: `(cpc_futuro == 1) OU (promessa_futuro == 1)`.
    - **Motivacao:** Esta combinacao garante que o modelo aprenda a priorizar telefones que nao apenas atendem (CPC), mas que tambem resultam em acoes produtivas (conversao). Como muitas vezes a promessa ocorre apos um CPC ja registrado no periodo, a operacao logica OR captura todos os eventos de sucesso do negocio.

2. **Impacto no Dataset:**
    - **Aumento de Eventos Positivos:** A inclusao da promessa aumenta ligeiramente a densidade de eventos positivos, o que ajuda o modelo a ter mais exemplos de "sucesso" para aprender, especialmente em casos onde o CPC pode nao ter sido marcado corretamente mas houve uma promessa.

## Consequencia
- O modelo agora otimiza diretamente para o sucesso financeiro/operacional (CPC + Conversao).
- O Baseline e as Metricas de Ranking foram atualizados para utilizar este novo alvo.
- O pipeline de Feature Engineering agora inclui esta etapa de consolidacao do label.
