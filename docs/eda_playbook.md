---
title: EDA Playbook for Financial Risk
owner: Data Science
effective: 2026-07-24
---

# EDA Playbook — Financial Risk

The checklist this project runs on every dataset and every refresh. It is deliberately
boring and deliberately ordered: the same moves, in the same sequence, so that nothing is
skipped because a chart looked interesting early.

Checks marked **[risk]** are specific to a rare binary target (fraud, default, churn).
They exist because the tools most data-science courses teach — correlation, scatter plots,
"remove the outliers" — quietly misbehave when the event rate is ~1%.

Implementation: `src/financial_risk/eda/profiling.py`. Every function returns a DataFrame,
so any check here can be written into a report or asserted on in CI.

---

## The same four moves in other stacks

This playbook is written in Python because that is what this project uses. The **moves** are
the transferable part — the tooling is a dialect. If you change employer you will very likely
change column, not row.

| Move | Python (here) | SQL / warehouse | R & SAS | Platform & BI |
|------|---------------|-----------------|---------|---------------|
| 1 First look | `df.info()`, `first_look()`, `ydata-profiling` | `COUNT(*)` / `COUNT(col)`, `information_schema` | `skimr::skim()`, `PROC CONTENTS`, `PROC FREQ` | dbt tests, Great Expectations, Soda |
| 2 Univariate | `describe()`, `quantile()`, seaborn | `PERCENTILE_CONT`, `NTILE`, `APPROX_QUANTILES` | `summary()`, `ggplot2`, `PROC UNIVARIATE`, `PROC MEANS` | Tableau / Power BI visual profiles |
| 3 Bivariate | `groupby().agg()`, `corr()`, `optbinning`, `scorecardpy` | `GROUP BY` segment rates | `scorecard` / `Information` packages, Enterprise Miner grouping | Databricks / Spark at scale |
| 4 Insight | Jupyter, matplotlib | the metric definitions the BI layer reads | R Markdown / Quarto, SAS ODS | Tableau, Power BI, Looker, Confluence |

Three things worth knowing:

- **SAS is not legacy trivia.** Regulated lenders and insurers still run it because its
  model-validation and audit trail is long-established. `PROC UNIVARIATE` and Enterprise
  Miner's interactive grouping may well be the working environment at a bank.
- **Past a few hundred million rows, EDA moves into SQL.** You cannot pull the table to a
  laptop, so the first look and the segment rates run in the warehouse. That is the SQL from
  week 5 doing EDA work.
- **The WOE/IV in move 3 is the same computation everywhere.** `optbinning` (Python), the
  `scorecard` package (R) and Enterprise Miner's grouping node all produce the table this
  playbook builds by hand.

---

## Move 1 — First look

*Before any chart. You cannot trust a plot of a column you have not inspected.*

| # | Check | Call | What would stop the project |
|---|-------|------|------------------------------|
| 1.1 | Shape, dtypes, distinct counts | `first_look(df)` | A numeric column parsed as text (week 4's `"12,000"`) |
| 1.2 | Missing cells per column | `first_look(df)` → `missing_pct` | A key feature 60% empty |
| 1.3 | **[risk]** Is missingness itself predictive? | `missingness_report(df, target)` | Different target rate when missing → keep an `is_missing` flag |
| 1.4 | Exact duplicate rows | `df.duplicated().sum()` | Double-counted transactions inflating every rate |
| 1.5 | Key uniqueness | `df[key].is_unique` | A "one row per transaction" table that isn't |
| 1.6 | **[risk]** Class balance | `df[target].value_counts(normalize=True)` | The number that governs every metric choice downstream |

**On missingness (1.3).** The reflex is to impute the median and move on. In risk work that
reflex destroys information, because *why* a value is absent is often tied to the outcome —
a blank income field on a credit application is rarely missing at random. Compare the target
rate when missing against when present; if they differ materially, add the flag as its own
feature, *then* impute.

---

## Move 2 — Univariate

*One column at a time.*

| # | Check | Call | Watch for |
|---|-------|------|-----------|
| 2.1 | Numeric summary | `df.describe()` | — |
| 2.2 | Mean vs median | `mean / median` ratio | On money, a ratio far from 1 means the mean is not "typical" |
| 2.3 | Distribution shape | histogram, raw **and** log | Heavy tails hide all structure on a linear axis |
| 2.4 | **[risk]** Outlier fences | `outlier_summary(df, cols)` | See the warning below before dropping anything |
| 2.5 | Categorical mix | `value_counts(normalize=True)` | The denominator behind every rate you later quote |
| 2.6 | Cardinality | `nunique()` | A 10,000-level "category" is an ID, not a feature |

**On outliers (2.4) — the most dangerous default in the toolkit.** Tukey's `1.5 × IQR`
fences are taught everywhere, usually followed by `df = df[~outliers]`. In fraud and claims
data the extremes are frequently *the event itself*. Measured on this project's demo sample:

| | share of rows | fraud rate |
|---|---|---|
| inside the fences | 90.9% | 0.77% |
| beyond the fences | 9.1% | 3.95% |

A one-line drop would have removed **~34% of all fraud** from the training data. Options, in
the order they are usually right here:

1. **Keep and transform** — a log transform compresses the tail and deletes nothing.
2. **Cap (winsorise)** — clip to a percentile so one extreme cannot dominate a coefficient.
3. **Drop** — only for *impossible* values (negative amounts, dates in 2190), not merely
   large ones. Cleaning should already have removed those upstream.

Whichever you choose, record the reason. "We excluded 9% of transactions" is a question you
will be asked in model review.

---

## Move 3 — Bivariate: every column against the target

| # | Check | Call | Reading |
|---|-------|------|---------|
| 3.1 | Segment rates | `groupby(seg)[target].agg(['count','mean'])` | Always report the **rate**, not the count |
| 3.2 | Numeric vs target | quantiles per class, boxplot | Distribution shifts a mean would hide |
| 3.3 | Correlation matrix | `df.corr()` | Redundant pairs and linear signal — **not** a verdict |
| 3.4 | **[risk]** WOE / Information Value | `rank_features_by_iv(df, target)` | The real feature ranking (see below) |
| 3.5 | **[risk]** Leakage screen | IV > 0.5, or "too good" features | Ask: is this known at scoring time? |
| 3.6 | **[risk]** Stability over time | `population_stability_index(dev, oot)` | Is the population still the one you modelled? |

### Why correlation is not enough (3.3 → 3.4)

Pearson correlation measures *linear* association. A threshold-shaped relationship — very
common in fraud — scores near zero. On this project's demo sample, one column:

| method | score | reading |
|---|---|---|
| Pearson correlation | 0.011 | looks useless |
| Information Value | 0.231 | **medium predictive** |

Same column (`amount_to_origin_balance`), opposite conclusions. Screening features on
correlation alone would have discarded one of the best signals available.

### Information Value bands

WOE bins a feature and measures how over- or under-represented the event is in each bin;
IV sums those contributions into one number per feature.

| IV | Reading |
|---|---|
| < 0.02 | unpredictive — candidate for removal |
| 0.02 – 0.1 | weak |
| 0.1 – 0.3 | medium |
| 0.3 – 0.5 | strong |
| **> 0.5** | **suspicious — check for leakage before celebrating** |

Two properties worth knowing:

- **A monotonic WOE across bins is what you want.** It means the feature behaves consistently
  across its range rather than by accident in one bucket.
- **IV is invariant to monotonic transforms.** `amount` and `log_amount` score identically,
  because IV is computed on quantile bins. You cannot inflate IV by taking a log — unlike
  correlation, which moves.

### The leakage screen (3.5)

An implausibly strong feature is a question, not a win. On this project `origin_new_balance`
scores IV > 2.0 — the account balance *after* the transaction. Fraud drains accounts, so a
post-transaction balance near zero nearly announces the label. Whether it is usable depends
on a question EDA cannot answer: **at the moment you must score, is this value known yet?**
If the model scores at authorisation, it is not, and shipping it buys a spectacular backtest
and a worthless live model.

Same instinct that rejected `is_flagged_fraud` in week 2.

### Stability / PSI (3.6)

```
PSI = Σ over bins  (actual% − expected%) × ln(actual% / expected%)
```

Bin edges come from the **development** sample, so the question is always "where did the new
traffic land relative to the population the model was built on".

| PSI | Action |
|---|---|
| < 0.1 | stable — no action |
| 0.1 – 0.25 | moderate shift — monitor, investigate the driver |
| > 0.25 | significant shift — investigate; likely refit |

Run per feature, on a regular cadence, against the development sample. Confirming stability
is most of what monitoring does; supervisory guidance on model risk expects deployed models
to be monitored on an ongoing basis, not validated once at build time.

**Rare-target trap.** On a ~1% event rate, a change from 0.97% to 1.27% reads as a 31% jump
but can rest on a couple of dozen events. Before escalating any rate change, count the events
behind it and test it — on this sample that particular difference gives a Fisher exact
p ≈ 0.37, i.e. noise.

---

## Move 4 — Insight

*A finding only counts once someone can act on it.*

| # | Check | Standard |
|---|-------|----------|
| 4.1 | Every claim leads with the business implication | "…therefore we should ___" |
| 4.2 | One number per sentence | A reader retains one; three retains none |
| 4.3 | Charts attached to claims | A chart without a claim is decoration |
| 4.4 | Scope disclosed | Sample size, date window, demo vs full data |
| 4.5 | Report regenerates from data | Numbers interpolated, never typed by hand |

---

## Before you leave EDA

- [ ] Every check in moves 1–3 has been run, and its result recorded — including the ones that passed.
- [ ] Any feature with IV > 0.5 has an explicit written answer to "is it known at scoring time?"
- [ ] Any row exclusion has a stated reason and a counted cost.
- [ ] The train/test split respects **time**, not random order.
- [ ] The metric choice reflects the class balance — not accuracy.
