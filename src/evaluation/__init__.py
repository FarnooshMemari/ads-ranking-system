"""Offline evaluation metrics for CTR prediction, ranking, and ads business KPIs.

- ``classification_metrics`` — CTR model quality (AUC, LogLoss, calibration)
- ``ranking_metrics`` — ranking quality (NDCG@K, Recall@K, MAP@K)
- ``retrieval_metrics`` — retrieval quality (Recall@K, Hit Rate@K, NDCG@K),
  implemented, cohort-aware (see docs/attribution_modeling_design.md §8)
- ``ranking_integration`` — the restricted retrieval-candidate ranking check
  (docs/ranking_integration_design.md Part B): never treats an unobserved
  campaign as a negative, reports intersection size before any metric
- ``ads_metrics`` — observed business metrics (CTR, CVR, ROAS)
"""
