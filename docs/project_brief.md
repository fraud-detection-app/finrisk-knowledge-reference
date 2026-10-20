---
title: Project Brief
owner: Data Science
effective: 2026-07-02
---

# Project Brief

## Business problem

Financial institutions need to prioritise suspicious transactions for analyst review. This
teaching project builds a fraud-risk assistant that ranks transactions, explains the key risk
drivers, answers project and model questions, and packages the work for an Australian data portfolio.

## Success criteria

- The data pipeline runs from the raw PaySim CSV to curated parquet.
- Feature engineering uses SQL window functions and documents its leakage boundaries.
- The model reports PR-AUC, recall, precision, a threshold, a model card and an error analysis.
- The assistant explains a scored transaction and answers knowledge-base questions with citations.
- The service runs locally and deploys to four AWS environments with a teardown path.

## Non-goals

- No real customer data.
- No production lending, credit, legal, or investment decisions.
- No always-on AWS infrastructure for the course.
