---
title: Frequently Asked Questions
owner: Data Science
effective: 2026-07-02
---

# FAQ

Q: Why use PR-AUC instead of accuracy?
A: Fraud is rare, so a model can achieve high accuracy by predicting non-fraud. PR-AUC focuses on
positive-class ranking performance.

Q: What is the AWS deployment?
A: The service runs as a container image on AWS Lambda (dev and sit) and on ECS Express Mode
(pre-prod and prod). Amazon Bedrock writes the explanations and Amazon S3 holds the model and the
knowledge index.

Q: How does a new policy document reach the assistant?
A: Through a pull request to the finrisk-knowledge repository. After review and merge, the ingest
pipeline builds a new index, checks it against the evaluation questions, and promotes it through
dev, sit, pre-prod and prod.
