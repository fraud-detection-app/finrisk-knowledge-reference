---
title: Model Card
owner: ML Engineering
effective: 2026-09-18
---

# Model Card

## Inputs

The fraud model uses the transaction amount, the transaction type, balance deltas, and time-safe
customer history features. History features only look at transactions that happened before the one
being scored.

## Target and excluded fields

The target is is_fraud. The is_flagged_fraud field is excluded from the model features because it
represents an existing business rule. Using it would leak the answer of an older system into the
model and inflate every metric.

## Metrics

Fraud is rare, so accuracy is misleading: a model that predicts "not fraud" every time is almost
always right. The model is judged on PR-AUC, which measures how well it ranks the rare positive
class, together with recall and precision at the chosen threshold.

## Limits

The data is synthetic. The model has never seen real customers, so its scores are for teaching and
must not be used for real decisions. High scores go to an analyst, see the fraud review policy.
