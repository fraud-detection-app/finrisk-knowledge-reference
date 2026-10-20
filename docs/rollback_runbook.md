---
title: Rollback Runbook
owner: Platform
effective: 2026-10-16
---

# Rollback Runbook

## Decide first

If an environment is unhealthy after a change, roll back first and debug afterwards. A rolled-back
release is a normal day. A prod that is debugged live is a story for the incident review.

## Roll back the application

Run the rollback workflow of finrisk-app with the environment and the commit SHA of the last good
release. It runs the same deploy steps with the older image, and the same approvals apply. The model
version and the knowledge index are not changed by an application rollback.

## Roll back the model

Run promote-model in finrisk-ml with the previous model version. The pointer in SSM changes and the
service reloads the old model within the pointer refresh interval.

## Roll back the knowledge index

Run promote-index in finrisk-knowledge with the previous index version. Every index version stays in
the knowledge bucket and is never overwritten, so an older version is always available. The workflow
writes the pointer for dev, sit, pre-prod and prod, and waits until the ready endpoint reports that version.

## Fix forward

A hotfix is a small pull request into main that goes through the normal pipeline. There are no hotfix
branches. After a rollback, revert or fix the bad change in git, otherwise the next merge ships it again.

## Freeze

While the cause is unknown, do not merge other changes into main.
