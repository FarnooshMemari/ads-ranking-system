"""Offline evaluation metrics for CTR prediction, ranking, and ads business KPIs.

- ``classification_metrics`` — CTR model quality (AUC, LogLoss, calibration)
- ``ranking_metrics`` — ranking quality (NDCG@K, Recall@K, MAP@K)
- ``retrieval_metrics`` — retrieval quality (Recall@K, Hit Rate@K, NDCG@K),
  implemented, cohort-aware (see docs/attribution_modeling_design.md §8)
- ``ads_metrics`` — observed business metrics (CTR, CVR, ROAS)
"""
