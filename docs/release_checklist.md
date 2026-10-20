---
title: Release Checklist
owner: Platform
effective: 2026-10-16
---

# Release Checklist

Run this list for every production release. Most lines are something to look at on a GitHub page.

## Before the pull request

- The CI checks are green.
- New behaviour is covered by a test.
- The VERSION file is bumped.
- No secrets are in the diff. The repositories are public.

## Before approving pre-prod and prod

- The image digest is the same in the dev, sit and pre-prod summaries.
- The dev and sit smoke tests are green.
- The model version in the summary is the one you meant, and its status is Approved.
- The index version in the summary is the one you meant.
- A rollback target is written down: the previous commit SHA, model version and index version.

## After the prod deploy

- The ready endpoint returns 200 and reports the digest you approved.
- One scoring call and one knowledge question work end to end.
- The release tag and the GitHub Release exist.
- No new error patterns appear in the CloudWatch log group for 15 minutes.

## If a line fails

Stop and follow the rollback runbook.
