---
title: Data Source - PaySim
owner: Data Engineering
effective: 2026-07-02
---

# Data Source: PaySim

## What PaySim is

PaySim is a synthetic financial transaction dataset used for fraud-detection teaching and research.
It simulates the transaction log of a mobile money service: transfers, payments, cash-in, cash-out
and debits. The data is generated from the statistical patterns of a real service, so it holds no
real customer records. That makes it safe to share publicly and to show in interviews.

## Why this project uses it

- The fraud use case is clear and common in banks.
- The fields are easy to explain to a beginner.
- The is_fraud label allows supervised learning.
- The full file is about 494 MB, large enough to need real data engineering but small enough for a laptop.

## Main fields

- step: the simulated hour of the transaction.
- type: CASH_IN, CASH_OUT, DEBIT, PAYMENT or TRANSFER.
- amount: the transaction amount.
- nameOrig, nameDest: the origin and destination account ids. They are used to aggregate history,
  never as raw model features, because an id has no stable meaning and lets a model memorise.
- oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest: balances before and after.
- is_fraud: the target label.
- is_flagged_fraud: the flag set by an existing rule. It is excluded from the model features to
  avoid data leakage: using it is like peeking at part of the exam answers.

## Data layers

- raw: the original CSV, never edited.
- bronze: the same rows with a standardised schema, stored as parquet.
- silver: cleaned data with nulls removed and types validated.
- gold: the features and the label, ready for modelling.

## Demo mode

DATA_MODE=demo runs the whole pipeline on a 5,000-row sample for fast iteration. Any other value
runs the full file through DuckDB without loading it all into memory.
