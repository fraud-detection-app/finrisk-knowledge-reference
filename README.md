# finrisk-knowledge

The knowledge base of the Financial Fraud Risk assistant, and the pipeline that turns it into a
search index. One of four repositories in the `fraud-detection-app` organisation:

| Repository | What it ships |
|---|---|
| `finrisk-infra` | Terraform: shared stack and the four environments |
| `finrisk-app` | the risk service (container image) |
| `finrisk-ml` | trained models |
| **`finrisk-knowledge`** (this one) | **the documents the assistant answers from, as a versioned index** |

What runs in an environment is **image digest + model version + index version**. This repo owns the
last one. A new or changed policy is a pull request: the review of that pull request is the compliance
sign-off. Nothing else changes what the assistant may say.

```
docs/*.md ──► ingest build ──► eval gate ──► publish ──► promote dev → sit → pre-prod → prod
 (PR review)  (incremental)    (recall@5)   indexes/<v>/   /finrisk/<env>/index  +  wait for /ready
```

## Layout

```
docs/                    the knowledge base: markdown with front matter (title, owner, effective)
ingest/                  documents -> index. No dependency on finrisk-app.
  docs.py                front matter, the document hash, lint
  chunking.py            cleaner (keeps _ in identifiers), sections, overlap, chunk ids
  embeddings.py          titan (Bedrock) | hash (offline stand-in) ; tfidf is fitted in build.py
  plan.py  build.py      the incremental plan and the build
  publish.py promote.py  S3 (never overwrite) and the SSM pointer (+ readiness poll)
  reader.py              a tiny search over an index (used by eval and the tests)
eval/                    questions.yaml, evaluate.py (recall@5, MRR, refusals, the gate), policy.py
tests/                   no network: fake S3, fake SSM, fake Bedrock
.github/workflows/       kb-ci.yml  ingest.yml  promote-index.yml
```

## Run it on your laptop

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                                             # no network, no AWS

python -m ingest lint                                        # front matter of every document
python -m ingest build --embeddings tfidf --out build/index  # a real index, no AWS
python -m eval.evaluate --index build/index                  # recall@5, MRR, refusals; exit 3 = gate failed

# incremental, offline: 'hash' vectors stand in for Titan (same reuse rules)
python -m ingest build --embeddings hash --out build/a
echo "A new sentence." >> docs/faq.md
python -m ingest diff  --embeddings hash --previous build/a    # the plan, nothing is built
python -m ingest build --embeddings hash --previous build/a --out build/b
#   docs: +0 ~1 -0 · chunks: reused 73, embedded 1, dropped 1
```

### Commands

| Command | What it does |
|---|---|
| `python -m ingest build` | documents -> index folder. `--embeddings titan\|tfidf\|hash` (or `$EMBEDDINGS`), `--previous <folder \| s3://… \| latest>` |
| `python -m ingest diff` | print `docs: +A ~C -D · chunks: reused R, embedded E, dropped X` without building |
| `python -m ingest publish` | upload to `s3://<kb>/indexes/<version>/`; refuses to overwrite; needs a passing `eval_report.json` |
| `python -m ingest promote --env E --version V` | write `/finrisk/E/index`, then poll `<app_url>/ready` for up to 10 minutes |
| `python -m ingest latest` | print the newest published version |
| `python -m ingest lint` | check the front matter |
| `python -m eval.evaluate --index DIR` | score the index and apply the gate |

## The incremental build

1. Read the previous index (the latest published one, or a folder): its manifest and chunks.
2. Compare each document's `sha256` with the previous manifest: added `+`, changed `~`, deleted `-`, unchanged.
3. Unchanged documents keep their chunk rows **and** their embedding rows. No Bedrock call.
4. Changed documents lose their old chunks; added and changed documents are chunked and embedded.
5. Deleted documents lose their chunks. Print `docs: +A ~C -D · chunks: reused R, embedded E, dropped X`.

Everything is rebuilt when the previous index cannot be mixed with this one (different model, dimensions
or chunk settings). **`EMBEDDINGS=tfidf` is always a full rebuild**: the IDF weight of every word depends
on the whole corpus, so changing one document changes the vector of every chunk. The build says so.

## The index format

This is the contract with `finrisk-app`. The app reads it; this repo writes it.

```
indexes/<version>/            version = YYYYMMDD-HHMMSS-<git sha7>
  manifest.json
  chunks.jsonl                one line per chunk
  embeddings.npy              float32 (chunks, dims); row i belongs to line i of chunks.jsonl        [titan]
  tfidf_vectorizer.joblib  +  tfidf_matrix.joblib                                                    [tfidf]
  eval_report.json            the gate's report (the app ignores it)
  build_plan.json             reused / embedded / dropped (the app ignores it)
```

`manifest.json` (uploaded last, so an index with a manifest is complete):

```json
{
  "version": "20261016-030405-ab12cd3",
  "created": "2026-10-16T03:04:05Z",
  "source_sha": "ab12cd34eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
  "embeddings": "titan",
  "embedding_model": "amazon.titan-embed-text-v2:0",
  "dims": 512,
  "docs": { "docs/datasource.md": { "sha256": "b90faaf7…", "chunks": 5 } },
  "chunks": 74,
  "chunking": { "max_words": 140, "overlap": 30 }
}
```

`chunks.jsonl` (keys in this order):

```json
{"id": "bc4777e5123ebbd6", "doc_path": "docs/datasource.md", "section": "Demo mode", "text": "Demo mode. DATA_MODE=demo runs …", "sha256": "35f9fb8e…"}
```

* chunk `id` = `sha256(doc_path + "\n" + text)[:16]`; chunk `sha256` = `sha256(text)`
* `doc_path` is the repo path of the document, e.g. `docs/fraud_policy.md`
* tfidf: `embedding_model` is `"tfidf"` and `dims` is the vocabulary size. The question is embedded by the stored vectorizer.

### Embedding the question (titan)

The service must embed every question with **the same request** the chunks were embedded with:

```python
bedrock_runtime.invoke_model(
    modelId="amazon.titan-embed-text-v2:0",
    body=json.dumps({"inputText": question, "dimensions": 512, "normalize": True}),
    contentType="application/json", accept="application/json")
# response body: {"embedding": [512 floats], "inputTextTokenCount": n}
```

Vectors are normalised, so cosine similarity is a dot product with `embeddings.npy`.

## The evaluation gate

`eval/questions.yaml` has answerable questions (each lists the documents that may answer it) and
`must_refuse` questions. An index may be published only if:

| Check | Rule |
|---|---|
| recall@5 | **>= 0.80** (an expected document is in the top 5 chunks) |
| no regression | recall@5 has **not dropped by more than 0.05** against the report of the index now in `/finrisk/prod/index` |
| refusals | every `must_refuse` question is refused, and no answerable question is |

MRR is reported, not gated. Exit code **3** means the gate failed. The report is written to
`eval_report.json` next to the index and published with it, so it becomes the next build's baseline.

The refusal patterns in `eval/policy.py` are **copied from finrisk-app**
(`rag/index.py` and the service's policy overlay). This repo does not import the app, so keep the two
in sync by hand.

## The workflows

| Workflow | Runs on | Does |
|---|---|---|
| `kb-ci.yml` (job `kb-check`) | every pull request, push to main | tests, front-matter lint, tfidf build from scratch, incremental plan against main's docs, eval vs main, job summary |
| `ingest.yml` | push to main touching `docs/**`, `ingest/**`, `eval/**`; manual | build (titan, incremental) → eval gate → publish → promote dev → sit → pre-prod → prod |
| `promote-index.yml` | manual: `version` | the promote chain for an existing version: a rollback is "promote an older one" |

`pre-prod` and `prod` are GitHub Environments with a required reviewer.

### Names the workflows need

| Kind | Name | Value |
|---|---|---|
| repository variable | `AWS_REGION` | `ap-southeast-2` (no account id in it, so it is a variable) |
| repository secret | `AWS_KB_BUILD_ROLE_ARN` | role `finrisk-kb-build` (OIDC, trusts `ref:refs/heads/main`) |
| repository secret | `KB_BUCKET` | `finrisk-kb-<account>-<region>-an` (holds the account id, so it is a secret) |
| environment secret (`dev`, `sit`, `pre-prod`, `prod`) | `AWS_ROLE_ARN` | role `finrisk-kb-promote-<env>` (OIDC, trusts `environment:<env>`) |
| environments | `dev`, `sit`, `pre-prod`, `prod` | deployment branch policy: `main` only; reviewer on `pre-prod` and `prod` |

The rule, the same in all four repos: a job **with** `environment:` reads the environment secret `AWS_ROLE_ARN`; a job **without** an environment reads a repository secret named after its role.

No access keys anywhere. Every job that logs in to AWS masks the account id right afterwards.

### Roles

| Role | Needs |
|---|---|
| `finrisk-kb-build` | read/write `indexes/*` in the KB bucket; `bedrock:InvokeModel` on the Titan embedding model; `ssm:GetParameter` on `/finrisk/prod/index` (the regression check reads prod's report; with no prod pointer yet the check is skipped, with a note) |
| `finrisk-kb-promote-<env>` | read the KB bucket; get/put `/finrisk/<env>/index`; get `/finrisk/<env>/app_url` |

## Rollback

Index versions are immutable. Run **promote-index** with an older version: it writes the same pointer
for each environment and waits until `/ready` reports that version.
