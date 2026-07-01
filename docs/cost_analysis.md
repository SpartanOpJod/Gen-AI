# Cost Analysis: LanceDB vs Pinecone

## Assumptions
- 100K, 1M, and 10M vectors.
- Small-to-medium workloads with occasional updates.

## Summary
- LanceDB is attractive for local/small deployments and low infrastructure cost.
- Pinecone is preferred for larger, more demanding production workloads that need managed scaling.

## Tradeoffs
- LanceDB: lower operational overhead, embedded local storage, but less managed scaling.
- Pinecone: higher cost, easier operational management, stronger enterprise features.
