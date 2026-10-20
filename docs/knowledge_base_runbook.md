---
title: Knowledge Base Change Runbook
owner: Compliance
effective: 2026-10-16
---

# Knowledge Base Change Runbook

## The rule

A new or changed policy document is a pull request to the finrisk-knowledge repository. The review
of that pull request is the compliance sign-off. Nothing reaches the assistant any other way.

## What the pull request checks

Every document needs front matter with a title, an owner and an effective date. The CI job builds
the index from scratch, runs the evaluation questions, and compares the result with main. It
fails when recall at 5 drops below 0.80 or falls by more than 0.05 compared with production.

## What happens after the merge

The ingest pipeline builds a new index version named by timestamp and commit. Only the documents
that changed are chunked and embedded again; unchanged documents keep their chunks and vectors.
The index is checked by the evaluation gate, published to the knowledge bucket, and promoted to dev,
sit, pre-prod and prod in that order. Pre-prod and prod need an approval.

## Deleting a document

Deleting a file in a pull request removes its chunks from the next index. The assistant stops
citing it once the new index version is promoted.

## Rollback

Index versions are immutable. To go back, run promote-index with an older version.
