# Retrieval Evaluation Report

## Metrics

- Recall@k: fraction of relevant chunks recovered in the top-k results.
- Hit Rate: whether at least one relevant chunk is returned.
- MRR: reciprocal rank of the first relevant hit.
- nDCG@k: ranking quality that rewards higher-ranked relevant chunks.
- Context Precision: proportion of retrieved chunks that are relevant.

## Output

The metrics are written to [evaluation/retrieval_results.json](../evaluation/retrieval_results.json).
