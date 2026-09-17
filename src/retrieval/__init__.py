"""Candidate generation / retrieval — narrows the full ad pool before ranking.

``cohort`` classifies users into cold/warm from train-period history only;
``candidate_generator`` implements the retrieval strategies themselves.
"""
