# Internal Legal Operations Assistant

A fictional ExampleCo RAG prototype. Employees ask documented procedural questions
through a Streamlit portal hosted on Replit. Answers cite evidence and pass validation, or the app
displays a Contact Legal fallback. It cannot approve or sign agreements, determine
legal rights, authenticate employees or automatically notify Legal.

## Current framework

The application uses hosted Pinecone for semantic search, local MiniLM embeddings,
BM25 keyword search, reciprocal rank fusion and LangGraph orchestration.
Use the `feature/pinecone` branch for this application.

The compatible index `legal-operations-minilm-vectors` is configured with 384
dimensions, cosine similarity and AWS us-east-1. The initial 49 fictional passages
were uploaded and their IDs verified. Startup, 16 retrieval smoke checks, six
complete-evidence retrieval checks and 91 local tests passed. A separate fresh 20-question Pinecone/Nebius
workflow evaluation was completed on 29 October 2026. Hosting on Replit is reported
by the project owner; independent hosted verification is not recorded.

The adapter accepts supplied dense vectors and validates the configured index.
Hosted embedding schemas are rejected because document and question vectors must
use the same MiniLM model and preprocessing.

## Hosted access

Hosting platform: Replit (reported by the project owner).
Published URL: **to be added**. Deployment type: **to be confirmed**.
Employees use the published URL; Preview is for development and `localhost:8501`
is for running on your own computer. Hosted keys/settings come from Replit
Secrets/Configurations; `.env` is for local development. Hosting does not add
employee authentication, document permissions or automatic Legal notifications.

## Setup

Run commands from the project root. Locally tested with Python 3.14; the owner confirmed Replit installation and
snapshot access. Independent published-deployment verification is not recorded. Direct dependencies are pinned;
transitive dependencies are not fully locked.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` only if you do not already have `.env`. Configure:

```dotenv
PINECONE_API_KEY=<private key>
PINECONE_INDEX_NAME=<compatible index name>
PINECONE_INDEX_HOST=<index host>
PINECONE_NAMESPACE=legal-demo
LLM_PROVIDER=nebius
NEBIUS_API_KEY=<private key>
LLM_BASE_URL=https://api.tokenfactory.nebius.com/v1
LLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
EVIDENCE_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
```

Never commit credentials. Replit reads the same variables through Secrets and
Configurations. An OpenAI-compatible SDK calls Nebius; that does not mean an
OpenAI answer model is used. Provider model availability can change.

## One-time Pinecone setup

The current local `.env` already points to the compatible index. To reproduce setup
in another environment, `python -m scripts.create_pinecone_index` creates the named
index if absent, verifies compatibility and updates only its name/host in local
`.env`. It never deletes indexes. Creation uses plan quota and may incur charges.
On Replit, copy the verified name/host to Configurations rather than relying on a
runtime `.env` file. Do not run resource creation on normal app startup.

## Index once, start without re-uploading

After configuring a compatible Pinecone index, the administrator uploads the
fictional corpus with:

```bash
python -m legal_assistant.knowledge_base
```

This sends vectors, text and metadata to Pinecone and can incur Pinecone charges.
It makes no Nebius calls. The index is validated before writing. Each corpus/model
snapshot has a deterministic namespace: the configured prefix plus a content hash.
After upload, record count and IDs must be visible before atomically switching
`storage/index.json`. Failed readiness preserves the previous local manifest.

On subsequent startup:

```bash
python -m scripts.start_app
```

Startup reconstructs the local manifest/BM25 corpus from bundled documents and
verifies that the matching Pinecone snapshot exists, then starts Streamlit on
0.0.0.0:8501. It never uploads vectors. Missing snapshots stop startup and require
an explicit administrator index operation. Alternatively prepare the manifest and
launch the local-only portal address:

```bash
python -m scripts.prepare_app
python -m streamlit run app.py
```

Changing eligible documents, dates or the embedding implementation requires indexing
again. New snapshots exclude removed documents without deleting older namespaces.
Old namespaces remain for rollback and may incur storage charges; delete them only
after checking that no deployment uses them. No scheduled refresh exists.

## Corpus, chunking and embeddings

`data/legal_faqs/`, `data/policies/`, and `data/reviewed_matters/` contain only
fictional English examples: 13 content files (12 JSON plus one text PDF), one PDF
metadata sidecar and 49 indexed passages. JSON uses a `records` array; sections can
inherit parent metadata. PDFs require a matching `.metadata.json` sidecar.

Only current approved, effective, non-expired sources with `synthetic=true` and
`access_scope=demo` are eligible. Reviewed matters additionally require closed,
legally reviewed, verified resolutions. Metadata markers do not prove real approval.
Text trimming and blank-page skipping are implemented; OCR and advanced table or
boilerplate cleanup are not.

Short FAQs stay intact. Policies split by section/sentence with 220-token maximum
and up to 40 tokens of sentence overlap. IDs and source/page metadata are retained.
MiniLM uses WordPiece tokens, 256-token inference truncation, masked mean pooling
and L2 normalization to produce 384-dimensional vectors. Questions and passages
use identical processing. ONNX Runtime runs local inference and tokenizers handles WordPiece processing.
The MiniLM ONNX model artifact is
SHA-256 verified and downloaded over HTTPS on first use. Verified existing local
assets can be reused. `MINILM_CACHE_DIR` optionally changes the cache location;
otherwise it is `storage/models/minilm`. This cache is not bundled or committed.

## Retrieval and answer workflow

BM25 supplies up to 12 positive keyword matches. Pinecone supplies up to 12 semantic
matches from the active namespace, filtered by organization, edition and demo
eligibility. Reciprocal rank fusion (sum of 1/(60+rank)) and conservative content
deduplication select four passages by default (`RETRIEVAL_LIMIT`, range 1–12).
Unknown remote IDs are discarded; the local manifest supplies trusted text/metadata.

LangGraph: Start → retrieve → grade evidence → generate draft → check citation IDs
→ validate factual claims → display answer/clarification, or Contact Legal.
A normal generated response uses three Nebius calls. Explicit human requests skip
search/models. Decisions depend on evidence suitability, applicability, consistency,
human judgment and claim support, not a numeric confidence cutoff. One bounded
claim validator has no correction/retry loop. Model assessments remain fallible.
Failed processing is recorded separately from insufficient evidence.

Each question has fresh state; displayed chat history is not supplied as model
memory. There is no checkpointer, reviewer resume or persistent chat audit log.
Legal escalation displays a fixed message and prepares local handoff details; it
does not send email or create a case. No employee authentication/permissions exist.
Use only fictional data until those controls and hosted-processing approval exist.

## Checks and evaluation

```bash
python -m unittest discover -s tests -t . -v
```

Regression tests mock Pinecone requests and make no Nebius calls. The local MiniLM
assets must be available or downloaded initially. Once a compatible snapshot is
indexed, retrieval checks use real Pinecone reads (no Nebius):

```bash
python -m evaluation.eval
python -m evaluation.eval_chunking
```

The main set has 20 questions: 16 expected answers and 4 escalations. Five additional
ambiguous scenarios are evaluation examples, not measured outputs. Earlier saved reports
remain historical; the fresh 29 October 2026 run is linked below.
95% faithfulness remains a target, not an established result. Live evaluation
(`python -m evaluation.resolution_eval --live`) incurs Nebius usage; the fresh run below used this workflow. Latency has no hard end-to-end deadline; SDK timeouts bound
individual network operations.

## Project structure

| Location | Responsibility |
| --- | --- |
| app.py / legal_assistant/app_pages/ | Portal navigation and chat |
| legal_assistant/knowledge_base.py | Eligibility, ingestion and snapshot publication |
| legal_assistant/chunking.py | Section/sentence splitting |
| legal_assistant/embeddings.py | Independent local MiniLM inference and token counting |
| legal_assistant/vector_store.py | Validated Pinecone connection and API adapter |
| legal_assistant/retriever.py | BM25/Pinecone filtering, fusion and deduplication |
| legal_assistant/bot.py / workflow.py | Model schemas, prompts, validation and graph routing |
| evaluation/ | Evaluation datasets, runners and reporting |
| tests/ | Offline regression checks |
| scripts/ | Preparation, startup and upload packaging |
| data/ | Fictional source documents |
| storage/ | Local manifest/model cache and saved evaluation artifacts |
| docs/ | Updated Pinecone guide/diagrams and framework-check summary |
| .streamlit/ / .replit | UI theme and deployment commands |
| Hosting: Replit | Streamlit, LangGraph, MiniLM inference and BM25; external Pinecone/Nebius services |

See [REPLIT_SETUP.md](REPLIT_SETUP.md). Build the upload ZIP after changes with
`python -m scripts.package_replit`. No Git push, deployment or automatic creation
of paid Pinecone resources is part of that command.

## Replit publishing environment

The UI dependencies are Streamlit 1.66.1 and Pillow 13.0.0. Replit's protected
system Python must not be modified with pip. `.replit` uses
`python -m scripts.build_environment` to create/bootstrap `.venv`, install the
pinned dependencies with user installs disabled and check compatibility. Its run
command is `.venv/bin/python -m scripts.start_app`. Base Python must include
`venv` and `ensurepip`; package firewall settings are preserved.
See REPLIT_SETUP.md for deployment secrets, port and access configuration.

## Current documentation and report review

The Word guide and diagrams in `docs/architecture/` now describe Pinecone.
See [Pinecone framework checks](docs/reports/Pinecone_Framework_Checks.html) for
recorded verification and historical report labels. Original resolution/validator
reports are unchanged historical baselines; the new live report is saved separately.
The refreshed chunking comparison includes Pinecone retrieval checks, but its
before baseline used a different corpus. No causal improvement claim is made.

## Latest full-workflow report

[Fresh HTML report](storage/evaluations/pinecone_full_workflow_20261029_232340/resolution_report.html)
([CSV](storage/evaluations/pinecone_full_workflow_20261029_232340/resolution_report.csv),
[JSON](storage/evaluations/pinecone_full_workflow_20261029_232340/resolution_report.json)).
All 20 cases completed: automated resolution 15/20 (75%), answerable resolution
15/16 (93.75%), expected escalations 4/4 (100%), one additional escalation and no
processing or quality-evaluator failures. Median/P95/maximum response times:
5.32/8.28/9.03 seconds. These are local-machine timings against remote services,
not Replit measurements. Automated development judgments are not independent
accuracy or employee-confirmed resolution. Review LEGAL-EVAL-15 for the extra
human-judgment escalation. Older reports remain historical.
