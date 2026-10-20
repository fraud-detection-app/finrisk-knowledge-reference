---
title: Deployment Runbook
owner: Platform
effective: 2026-10-16
---

# Deployment Runbook

## Branching

The project uses trunk-based development. Work happens on short-lived feature branches that go
through a pull request into main. Main is always releasable. There are no environment branches:
dev, sit, pre-prod and prod are stages of one pipeline run.

## Environments

- dev and sit run on AWS Lambda with a Function URL. They scale to zero.
- pre-prod and prod run on ECS Express Mode. They stay warm and cost money, so tear them down after class.

## How a change reaches production

1. Open a pull request. The CI checks must pass before it can merge.
2. Merging to main builds the artifact once and names it after the commit SHA.
3. The same artifact is deployed to dev, then sit, then pre-prod, then prod. Pre-prod and prod wait for a reviewer to approve.
4. Each deploy runs a smoke test: the health endpoint must report the expected image digest, then the ready endpoint must return 200.
5. After prod succeeds, the pipeline tags the release from the VERSION file.

## Timing

A Lambda deploy takes about a minute and a half. An ECS Express service took about five to six minutes
to come up the first time in the rehearsal of 17 October 2026 (pre-prod 5 min 39 s, prod under 4 min),
so approve pre-prod early and keep working while it starts.

## Who deploys what

The application image, the model version and the knowledge index each have their own pipeline. A
new policy document does not need an application deploy, and a new model does not need an index build.
