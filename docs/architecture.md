---
title: Architecture Overview
owner: Platform
effective: 2026-10-16
---

# Architecture Overview

## Data flow

PaySim CSV files land in the raw layer. They are standardised into bronze parquet, cleaned into
silver, and turned into gold features with DuckDB SQL. The fraud model trains on the gold table and
is registered in the SageMaker model registry. The risk service scores a transaction, explains the
score with SHAP reasons, and asks Amazon Bedrock (DeepSeek v3.2) to write the narrative.

## What runs in an environment

An environment (dev, sit, pre-prod, prod) runs three independent versions: the application image
(identified by its digest), the model version and the knowledge index version. Each has its own
repository and pipeline. The service reads the model and index versions from per-environment SSM
parameters and reloads them without a redeploy.

## Four repositories

- finrisk-infra: Terraform for the shared stack and for each environment.
- finrisk-app: the risk service and its container image.
- finrisk-ml: training in SageMaker, the evaluation gate, model approval and promotion.
- finrisk-knowledge: these documents, the ingest pipeline and the retrieval evaluation.

## Environments and accounts

Large companies use one AWS account per environment. This course uses one account and separates the
four environments by the name prefix finrisk-dev, finrisk-sit, finrisk-pre-prod and finrisk-prod,
with one Terraform state and one IAM role per environment.
