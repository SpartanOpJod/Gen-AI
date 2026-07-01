# Cost Comparison: LanceDB vs Pinecone Serverless

## Assumptions
- Pricing is estimated from public cloud patterns and assumes standard vector indexes.
- 100K, 1M, and 10M vectors are used as the comparison points.
- The comparison is for approximate similarity search workloads with moderate update frequency.

## Comparison

| Scale | LanceDB | Pinecone Serverless | Notes |
| --- | --- | --- | --- |
| 100K vectors | Lowest operational cost; local embedded deployment | Higher managed-service cost | LanceDB is more cost-effective for small workloads |
| 1M vectors | Low cost with local storage, but scaling becomes more operationally involved | Higher cost with easier scaling | Pinecone is attractive when ops burden matters |
| 10M vectors | Cost-effective only if self-managed infrastructure is acceptable | Better managed scaling and operational simplicity | Teams often migrate back to managed DBs here |

## Tradeoffs
- LanceDB: lower cost, simple deployment, weaker managed operational guarantees.
- Pinecone Serverless: more expensive, but better managed scaling, reliability, and enterprise support.
