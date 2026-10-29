# Internal Legal Operations Assistant

A local Streamlit prototype that helps employees find documented legal procedures and policies. The ExampleCo dataset is entirely fictional and must not be treated as actual company policy or legal advice.

## Run

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m legal_assistant.knowledge_base
python -m streamlit run app.py
```

Open http://localhost:8501. The legal sample index is already built. Restart Streamlit after code/configuration changes, and refresh the search index after changing documents.

The app opens on a sample Legal operations portal. Select **Open Legal assistant** or the **Legal assistant** navigation option to chat; **Overview** returns to the portal. Conversation display is retained when navigating within the same session. The landing page lives in `legal_assistant/app_pages/home.py`, chat UI in `legal_assistant/app_pages/assistant.py`, and navigation in `app.py`. The native theme is configured in `.streamlit/config.toml`. The portal uses a navy sidebar and a wide overview with six topic shortcuts. Topic shortcuts prefill the assistant input without submitting a question; users review and send it themselves. Chat uses a centered reading layout, and index maintenance is performed from the terminal with `python -m legal_assistant.knowledge_base`; no refresh button is shown to employees.

## Models and configuration

Keep credentials in `.env`, never in chat or Git:

```dotenv
LLM_PROVIDER=nebius
NEBIUS_API_KEY=your_key_here
LLM_BASE_URL=https://api.tokenfactory.nebius.com/v1
LLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
EVIDENCE_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
LLM_TIMEOUT_SECONDS=30
LLM_MAX_RETRIES=0
RETRIEVAL_LIMIT=4
CHROMA_COLLECTION=legal_operations
```

MiniLM embeddings and Chroma/BM25 search run locally. Grading and answer generation send the question and retrieved passages to Nebius and incur provider usage. The installed `openai` package is used as a compatible client; no OpenAI credential is used for Nebius.

## Dataset

- `data/legal_faqs/`: employee questions and answers.
- `data/policies/`: illustrative approved procedures for NDAs, contract intake, signature authority, privacy review, incident routing, legal holds, employment concerns, disputes, and vendor onboarding.
- `data/reviewed_matters/`: sanitized fictional historical examples; these do not establish policy, privilege, approval, or precedent.

JSON documents contain a `records` array. Preserve source IDs, organization, policy_version, jurisdiction, effective_from, updated_at, document_status, approval_status, access_scope, and synthetic metadata. Ingestion excludes superseded, future-effective, expired, or unapproved records. Historical matters must also be closed, legally reviewed and resolution-verified. Reindex when dates/statuses change.

This prototype deliberately accepts only `synthetic=true` and `access_scope=demo` records. It cannot authenticate employees or enforce real employee/group permissions. Do not relabel real confidential documents as demonstration data. Company sign-in, retrieval permission enforcement in both branches, source-system access, retention/audit controls, and approval of hosted processing are required before introducing real internal documents. No authentication or confidential-document support is claimed.

## Workflow

Retrieve using Chroma semantic search and BM25 keyword search, filtered by organization and policy edition. Combine rankings using reciprocal rank fusion. Then:

**LangGraph: Start → Retrieve → LLM-grade evidence → Generate draft → Validate citation IDs → Validate claim support → End, or Escalate → End.**

The grader checks relevance, applicability, conflicts, and full/partial/no coverage. It selects supporting passages. The answer model receives those passages and coverage gaps. Partial answers cite supported information and identify missing details; the assistant must not invent policy, approve/sign agreements, determine legal rights or enforceability, or promise outcomes. Current policies take priority over historical examples. Human decisions and unresolved conflicts go to Legal. Evidence assessments remain internal. The UI does not display an assessment table or retrieved passages; inline answer citations remain visible.

There is now one bounded post-generation claim-support check before release, in addition to mechanical citation-ID validation. Both model grading and claim assessment are fallible; the 95% faithfulness target is not established. Each question is independent; chat history is not used to resolve follow-up references.

## Legal handoff

The UI displays a contact-Legal message only on escalation or an explicit employee request. There are no persistent Legal-contact or download buttons. A handoff packet is retained internally for a future integration; nothing is sent to Legal. Connect an approved internal intake/case-management system with successful-delivery receipts before claiming a live handoff.

## Checks

```bash
python -m unittest discover -s tests -t . -v
python -m evaluation.eval
python -m evaluation.eval_chunking
# Optional: billed provider calls for legal development cases
python -m evaluation.eval --llm
```

Default evaluation checks retrieval only and skips cases requiring model routing. These are development examples, not a held-out legal accuracy benchmark. Legal reviewers must approve a representative test set before employee rollout.

## Project structure

```text
ProjectWeek2/
├── app.py                 # Streamlit entry point
├── legal_assistant/       # Ingestion, chunking, retrieval, model calls and graph
│   └── app_pages/         # Overview and assistant UI
├── evaluation/            # Evaluation questions, runners and reports code
├── tests/                 # Regression tests
├── scripts/               # Preparation, startup and upload packaging
├── data/                  # Fictional FAQs, policies and reviewed matters
├── storage/               # Generated index and evaluation artifacts
├── docs/                  # Learning guides and architecture images
├── .streamlit/            # Portal theme
├── .replit                # Replit build, run and port settings
├── requirements.txt       # Pinned direct dependencies
└── .env.example           # Credential-free settings template
```

Run module commands from the project root using `python -m ...`; do not execute
nested Python files by their filesystem paths. Existing Word guides and saved
reports are historical snapshots and may show the previous file locations.


| File | Objective |
| --- | --- |
| app.py | Shared portal branding, session initialization, and page navigation |
| legal_assistant/app_pages/home.py | Overview and topic shortcuts that prepare questions |
| legal_assistant/app_pages/assistant.py | Chat display, policy-edition selection, Back/Clear actions, and local handoff presentation |
| legal_assistant/bot.py | Reusable evidence grading, grounded answers, citations, and fallback helpers |
| legal_assistant/workflow.py | LangGraph state, nodes, conditional routes, and single-request execution |
| legal_assistant/knowledge_base.py | Configuration, JSON/PDF validation and filtering, chunk metadata, and indexing |
| legal_assistant/chunking.py | MiniLM token counting, section detection, sentence-aware splitting and overlap |
| legal_assistant/retriever.py | Keyword/semantic search and result fusion |
| evaluation/eval.py / evaluation/eval_set.py | Legal development checks and questions |
| evaluation/eval_chunking.py | Complete-evidence retrieval checks and saved before/after comparison |
| evaluation/benchmark_validation.py | Billed development benchmark for validator latency and faithful/unfaithful drafts |
| tests/test_bot.py | Source filtering, grading, model handling and fallback tests |
| tests/test_app_ui.py | Headless checks for chat navigation, missing-index guidance, human routing, and sanitized failures |
| tests/test_chunking.py | FAQ integrity, section boundaries, sentence overlap, token limits, and metadata |
| tests/test_claim_validation.py | Claim assessment, complete output checks, bounded call settings and release routing |
| requirements.txt | Python dependencies |
| .env / .env.example | Private local configuration / safe template |
| .gitignore | Exclude credentials, data and runtime artifacts |
| data/ | Fictional legal input documents only |
| storage/ | Generated index and manifest |
| .venv/ | Project environment |

Text-based PDF and JSON ingestion are implemented. DOCX parsing, scanned-PDF OCR, and company document connectors are not yet connected.

## Fictional dataset scope

The demo covers all three topic groups: contracts/NDAs, general internal legal operations FAQs, and privacy/vendor onboarding. It contains 9 FAQs, 30 JSON policy records/sections across ten files, one policy PDF, and 4 reviewed-matter examples. All records are synthetic and marked for demonstration only.

## Fictional PDF example

`data/policies/fictional_ai_tools_policy.pdf` is a one-page fictional External AI Tools Policy. It is paired with `fictional_ai_tools_policy.metadata.json`, containing its source ID, organization, edition, effective date, approval status, and demo-only markers. The sidecar is metadata, not a separate document.

`legal_assistant/knowledge_base.py` uses `pypdf` to extract each page before applying section- and sentence-aware chunking. Each passage preserves the filename, original source ID, and one-based page number. For example, `POL-PDF-001:p1:0` refers to the first passage on page 1. Chunk suffixes now identify sequential chunk numbers, not character offsets; old citations may refer to the previous index. The same eligibility rules apply to PDF and JSON sources. PDFs without sidecars fail indexing with a missing-metadata error. Encrypted PDFs and files without extractable text are rejected; OCR is not implemented. Blank pages are skipped.

Try: **Can I paste an unsigned contract into an unapproved external AI tool?** The retrieved evidence should include `POL-PDF-001` and page 1. This tests a fact unique to the PDF. The old JSON dataset remains unchanged; the original combined index contained 26 passages, the additional JSON policies increased it to 44, and the initial structure-aware index contained 46 passages; the expanded clarification-policy index now contains 49 passages.

## LangGraph migration: stage 1 implemented

`legal_assistant/workflow.py` compiles a `StateGraph` with retrieval, grading, generation, citation validation, claim-support validation, and escalation nodes. `legal_assistant/app_pages/assistant.py` invokes `run_workflow()`; live evaluation uses the same graph. `legal_assistant/bot.py` exposes `grade_evidence()` and `generate_answer()` with all routing owned by `legal_assistant/workflow.py`. The former `grade_and_answer()` compatibility helper and unused assessment-table formatter have been removed; model-call tests now exercise the graph.

The graph preserves full/partial evidence answers, clarification text, source filtering, citation checks and local-only handoff preparation. Explicit human requests bypass search/models. Empty retrieval, failed grades, errors, model-requested escalation and invalid citations route to the escalation node. Raw exceptions/provider payloads are not stored in request state. The draft now passes through one claim-support validation call before display; there are no correction or regeneration loops.

Each invocation has fresh state and a bounded graph step count. There is **no checkpointer, SQLite database, thread memory, interrupt or reviewer simulation yet**. A clarification is a terminal response for now. Graph branch tests run in `tests/test_bot.py`; multi-turn acceptance scenarios are not implemented. LangGraph installs core ecosystem dependencies; this is not a migration of retrieval/model calls to LangChain chains. Tracing is not enabled by this change.

Restart Streamlit after upgrading dependencies and changing modules. No dataset or model changes, and no index rebuild, are required for this stage.

## Internal evidence assessments

Evidence confidence is an uncalibrated model assessment of source support, not answer accuracy. Assessments and processing status remain internal; the user-facing assessment table has been removed. Routing still checks relevance, coverage, applicability, conflicts, human judgment and valid citation IDs.

## Performance instrumentation and resource reuse

Each LangGraph result includes elapsed seconds for retrieval, grading, generation, validation/escalation when run, and total request time. Attempted failed stages are timed too; skipped stages are absent. Timings are retained internally and are not shown in the sidebar. Timings include retries and provider/network waiting; they are not model-only inference times.

A bounded process-local cache retains the latest two manifest snapshots, their Chroma/embedding resources and per-organization/per-edition BM25 indexes. An atomic index refresh or metadata change automatically selects a fresh resource. Questions, answers and per-request graph state are not cached. Shared local searches use a lock; permission/scope filters remain in both retrieval branches. The API client is reused by provider/endpoint/key/timeout/retry settings and replaced when that configuration changes. Restart the process after editing .env.

Grading explanations and answers are instructed to be concise. GRADE_MAX_TOKENS defaults to 700 and ANSWER_MAX_TOKENS to 500. These can be overridden in .env. If a structured response is truncated, it fails closed rather than displaying a partial answer; increase the appropriate limit if needed. Routing rules and citation checks remain in place. Streaming is not implemented.

## Response-time settings

The evidence grader and answer model now use `Qwen/Qwen3-30B-A3B-Instruct-2507`. They remain independently configurable through `EVIDENCE_MODEL` and `LLM_MODEL`. Both models must support structured JSON output.

`RETRIEVAL_LIMIT=4` sends up to four distinct passages after BM25/Chroma rank fusion. Exact duplicate content is removed only within matching organization, edition, jurisdiction, effective-date and status scopes. Historical matters remain distinct from current guidance; contradictory text is retained. This reduces prompt size but may reduce coverage on broad questions. Increase the limit up to 12 if evaluation identifies missing evidence. No reindex is needed.

`LLM_TIMEOUT_SECONDS=30` and `LLM_MAX_RETRIES=0` replace the previous 60-second request timeout and automatic retry. These settings avoid repeated waits on a slow provider; they do not accelerate provider inference or impose a 30-second total workflow deadline. Grading, generation, and claim validation are separate requests, and SDK timeouts apply to network operations. Failed requests still produce a safe Legal-contact fallback rather than fabricated answers. Restart Streamlit after changing `.env`.

A limited live check on the fictional dataset returned two cited procedural answers in 2.5–2.8 seconds and escalated two unsupported/human-judgment questions in about 1.1 seconds. The previous 235B answer model timed out at 30 seconds in the comparison request. These single-run observations are not production latency guarantees or measured accuracy. The smaller model initially omitted inline citations; the prompt now explicitly requires them, and validation still rejects invalid answers. The historical experiment files were removed during project cleanup. Broader reviewer-approved evaluation remains necessary.

## Additional fictional policy documents

Six JSON documents in `data/policies/` add 18 independently retrievable sections under the existing ExampleCo (fictional) / 2026.1 scope:

| File | Coverage | Source IDs |
|---|---|---|
| `nda_request_and_review.json` | NDA intake, counterparty edits, executed agreement handover | POL-NDA-010–012 |
| `confidential_information_handling.json` | Need-to-know access, external disclosure, accidental disclosure | POL-CONF-010–012 |
| `privacy_review_and_data_minimization.json` | Privacy intake, minimum necessary data, purpose/location changes | POL-PRIV-010–012 |
| `privacy_retention_and_requests.json` | Retention/deletion, employee requests, vendor data incidents | POL-DATA-010–012 |
| `vendor_due_diligence.json` | Procurement intake, Security package, contract/data terms | POL-VEND-010–012 |
| `vendor_changes_and_offboarding.json` | Readiness checks, renewals/changes, offboarding | POL-VEND-020–022 |

All approval labels describe fictional sample policies only. No actual company approval, vendor clearance, retention deadline, or Legal-system integration is asserted. Refresh the index after editing these documents. Six additional retrieval examples are included in `evaluation/eval_set.py`; default evaluation checks retrieval only.

## Current chunking and how to test it

`legal_assistant/chunking.py` replaces the former 600-character windows/500-character stride. Short FAQ question/answer pairs are preserved verbatim. JSON records and explicit `sections` are kept separate; missing section IDs are generated from the parent ID, while source identity and organization/edition metadata are inherited. PDF pages remain separate for reliable page citations. Markdown headings, short numbered headings, uppercase headings, and paragraph boundaries are used where detectable. This is a heuristic English text splitter, not a PDF layout engine or a legal parser; unmarked headings and complex layouts are not guaranteed to be detected. Sections spanning PDF pages are not automatically joined. OCR remains unsupported.

Each passage is limited to 220 tokens including special tokens and any repeated heading, counted with Chroma MiniLM's own local WordPiece tokenizer with truncation and padding disabled for counting. The installed embedding tokenizer truncates at 256 tokens, so this avoids silently losing the tail of an indexed passage. An oversized section is packed at complete sentence boundaries; up to 40 tokens of complete trailing sentences are carried into the next passage when space permits. No overlap is carried between unrelated sections. Long FAQ answers repeat a reasonably sized question. Oversized sentences/headings use a word/token-offset fallback and are marked `split_fallback=true`; preserving complete sentences is impossible when one sentence exceeds the token budget. Abbreviation handling is heuristic.

Metadata now records source file/format, section ID/title, chunk index, token count, chunking version, and fallback status, in addition to existing organization, edition, status, dates and PDF page. Chunk IDs can change on rebuild, so rerun old questions for current citations. Rules and exceptions that fit together stay together more often; no splitter guarantees retrieval of every distant exception. Explicit source sections and retrieval evaluation remain essential.

The refreshed index contains 46 passages. All 14 existing retrieval checks pass. Six additional checks require specified rule/condition phrases to appear together in at least one of the top four retrieved passages: **5/6 before, 6/6 after** on the same corpus and retrieval settings. These are hand-written development examples, not independent accuracy, faithfulness or production benchmarks. Before/after details are in `storage/chunking_comparison.json`. The sentence/boundary tests also include a constructed long-policy example so the oversized-section behavior is tested even though the current sample sections are short.

To reproduce checks without API calls:

```bash
.venv/bin/python -m unittest tests.test_chunking tests.test_bot tests.test_app_ui -v
.venv/bin/python -m evaluation.eval
.venv/bin/python -m evaluation.eval_chunking
```

In the UI, clear the conversation and ask each question separately. Look for both the useful guidance and the restriction, with valid source citations:

| Question | Expected evidence/behavior |
|---|---|
| Does approval for a different AI tool or purpose approve my proposed use? Is a turnaround guaranteed? | State that previous approval does not cover the proposed use, the assistant cannot verify current approval, and no approval date/turnaround is guaranteed. Cite POL-PDF-001, page 1. This was the incomplete-passage retrieval case before the change. |
| What information do I submit to request review of an AI tool, and may I upload restricted information while review is pending? | Include intake details and the prohibition on uploading restricted information during review; cite POL-PDF-001. |
| Is an NDA alone enough to authorize employee-data sharing with a vendor? | No. Explain required Privacy review and approvals; an NDA alone does not authorize sharing. |
| Can vendor-held employee data be deleted when an active legal hold covers the records? | Preserve covered records and contact the Legal hold owner before deletion. |
| Approve sharing this employee spreadsheet with Vendor X now. | Redirect to Legal; improved chunking does not grant the assistant approval authority. |

User-facing responses still depend on the grader and answer model, so wording may vary or generation may fail. The checks above establish evidence retrieval and tested software behavior; they do not verify every generated claim. Restart Streamlit to reload the new module before testing; the index has already been rebuilt.

## Live claim-support validation

Every generated answer or clarification now receives one post-generation claim-support check before display. Direct human requests, rejected evidence, empty retrieval, explicit model escalation and invalid citation IDs bypass this extra call. There is no FAQ shortcut, retry/correction loop, automatic human notification or new assessment UI.

`bot.validate_claim_support()` sends only the question, draft (numbered nonempty lines), and the exact selected passages used for generation. It does not provide the grader's confidence as proof. The model extracts atomic claims and labels them supported, unsupported or contradicted with evidence IDs and brief explanations. Each nonempty line must be accounted for as claims or pure nonfactual text. Code verifies coverage and evidence references; the model must still identify all claims within a line and assess their meaning correctly. This is not a guarantee that no claim was overlooked. Citation support within a line is also a model judgment, not a formal proof.

The graph releases a draft only if the assessment is complete, all assessed factual claims are supported, references are valid and no actual approval/authority decision is being granted. A substantive answer without assessed claims is rejected. A pure clarification may have no factual claims and no numerical score. Unsupported claims, contradictions, incomplete/invalid output, refusals, truncation and validation errors withhold the draft and use the Legal-contact fallback. A failed validator call is recorded as a processing failure, not low evidence support. The unapproved draft remains internal and is not displayed.

Configuration (defaults work without editing credentials):

```dotenv
# Blank/unset uses EVIDENCE_MODEL; configure a separately evaluated model if desired.
CLAIM_MODEL=
CLAIM_TIMEOUT_SECONDS=10
CLAIM_MAX_TOKENS=1600
```

The claim call overrides SDK retries to zero, regardless of other stage settings. The timeout is an SDK network timeout, not a hard whole-workflow deadline. It accepts values up to 30 seconds. Output is capped at 16 claims/1600 tokens by default; if assessment cannot fit, it fails closed. Using the same model for generation and validation can repeat errors; independent reviewer calibration is still necessary.

Graph state includes `claim_validation`, `faithfulness_percent` and `timings["claim validation"]`. The percentage is computed over the model's assessed claims only when coverage/reference checks pass, is null for incomplete/failed/no-claim assessments, and is not an independently measured accuracy or a 95% routing threshold. Release requires all assessed claims supported; a known unsupported approval claim cannot pass on a high average. There is still no persistent logging of every employee request.

To repeat the development benchmark (billed provider calls):

```bash
.venv/bin/python -m evaluation.benchmark_validation --live
```

It runs three fictional questions through the actual graph, a direct-human request, and seven deliberately faithful/unfaithful draft fixtures. It saves the exact evidence, drafts, validation decisions, model/index version information and timings to `storage/evaluations/claim_validation_benchmark.json`. No evaluation-report UI has been added. The dataset is a small development sample, not held-out or Legal-reviewed.

In the initial live run, all three generated answers were released; validator overhead ranged from **1.7 to 7.8 seconds** (median **2.1 seconds**). Total request times were **3.8, 5.4 and 10.4 seconds**; the longer cross-policy NDA/Privacy answer required the longest check. The same-run total excluding the validator stage is a timing decomposition, not a separate randomized baseline. All seven fixture accept/reject decisions matched expectations, including invented deadlines, broadened NDA approval, contradicted legal holds, unauthorized approval, faithful paraphrase and clarifications. These observations do not establish production latency, evaluator reliability, legal correctness or the 95% faithfulness target. A held-out reviewer-approved evaluation remains necessary.

Restart Streamlit to reload modules. No index rebuild or dependency installation is required for this change.

## Twenty-query end-to-end resolution evaluation

`evaluation/eval_set.py` now defines exactly 20 unique development questions: 16 expected answers and 4 expected escalations. Every answerable question has explicit required-guidance criteria authored from the fictional corpus; escalation questions record their expected reason. The set includes paraphrased procedures, policy/PDF questions, a multi-policy NDA/Privacy question, a historical reviewed matter, unsupported content, a legal interpretation, an actual approval request and a direct human request. It is not a held-out or independently Legal-reviewed benchmark.

Run the complete workflow with billed provider calls:

```bash
.venv/bin/python -m evaluation.resolution_eval --live
```

The default writes a timestamped folder under `storage/evaluations/`; `--output PATH` selects a folder explicitly. JSON, CSV and standalone HTML reports are saved progressively. Every case invokes the real `run_workflow()` once, with its normal early-exit routes; returned answers then receive a separate offline quality assessment against required guidance and exact selected evidence. This extra evaluation call is not part of employee chat. It uses `EVAL_MODEL` if set, otherwise the configured claim/grade model, with a 15-second SDK timeout and zero retries. No production UI or retrieval/answer behavior is changed.

The reports record question and case ID, expected action/outcome/guidance, actual action and answer, complete/partial/incorrect/unassessed/not-answered quality, escalation and error reasons, per-stage/total response latency, offline quality-evaluation time, evidence, draft and live claim assessment. Exact expected-source retrieval is tracked separately; a supported alternative source is not automatically an answer-quality failure. Quality assessment errors leave a case unassessed and are separately counted, never silently passed.

Definitions:

| Metric | Calculation |
|---|---|
| Automated rubric-verified resolution rate | Released answers passing live claim validation and all required-guidance checks / all evaluated questions |
| Answerable-question resolution rate | Same resolved answers / questions expected to be answerable |
| Correct escalation rate | Non-error escalations for expected-escalation questions, excluding failed citation/claim checks / expected-escalation questions |
| Unnecessary escalation rate | Non-error escalations of expected-answerable questions / expected-answerable questions |
| Processing failure rate | Workflow errors or failed stages / all evaluated questions; offline evaluator errors reported separately |
| Correct handling rate | Resolved answers plus correctly escalated cases / all queries |
| Median / P95 latency | Full workflow wall-clock time, excluding the separate offline assessment; P95 uses nearest rank |

These are automated development assessments, not user-confirmed first-contact resolution or independent ground truth. The same provider/model family can be involved in generation, live validation and offline assessment. Correct escalation is action-based handling with error/validation guards; the recorded rationale remains available for review and is not independently verified. The run does not establish 95% faithfulness or production latency.

The initial saved run is in `storage/evaluations/resolution_20_queries/`. All 20 queries ran: 16 answers met every automated guidance check, 4 expected escalations occurred, and no workflow/evaluator failures occurred. Automated resolution was 16/20 (**80%** overall), answerable resolution 16/16 (**100%**), correct escalation 4/4 (**100%**), unnecessary escalation **0%**, median response **4.83 s**, P95 **7.54 s**, maximum **8.57 s**. The four intentional human/out-of-scope cases explain why the overall automated resolution rate is 80%, even though all expected paths matched.

The existing `evaluation/eval.py` remains a retrieval/routing smoke runner; its default now checks 16 answer-source cases and skips 4 escalation questions. Use `evaluation/resolution_eval.py --live` for the full resolution report. `tests/test_resolution_eval.py` adds six no-network regressions for dataset uniqueness, resolution guards, error classification, metric denominators/P95, quality-reference validation and report exports.

## Grouped dataset and saved-result review (no new provider calls)

`evaluation/scenario_eval_set.json` is an evaluation-only dataset outside `data/`: 16 expected-answer questions, five ambiguous questions and four expected Legal redirects. It includes illustrative expected responses, expected handling, and prohibitions for ambiguous cases. Expected examples are authored guidance, not captured outputs. The original 20-case evaluation and its report remain unchanged.

Three fictional policy files were added under `data/policies/`: `nda_approval_scope.json`, `privacy_sharing_context.json`, and `vendor_change_context.json`. Their source IDs are POL-CLAR-NDA-001, POL-CLAR-PRIV-001, and POL-CLAR-VEND-001. They clarify missing approval/scope information without granting approvals or changing the existing policy edition. The local index was rebuilt and now contains **49 passages** from 43 JSON records/sections plus the PDF (13 content files: 12 JSON files and one PDF).

Generate the grouped review from saved results only:

```bash
.venv/bin/python -m evaluation.grouped_eval_report
```

This script makes no provider calls and does not execute the RAG workflow. It writes HTML, CSV and JSON to `storage/evaluations/grouped_review/`, matching saved baseline results by case ID and exact question. Original results show 16/16 expected answers resolved by the automated rubric, 4/4 expected escalations, 80% overall automated resolution, median 4.83 s and P95 7.54 s. All five ambiguous scenarios are explicitly **not run**: actual actions, responses, success rates and latency are unknown, not zero or assumed successful.

The report displays both the baseline and current index fingerprints. The saved results predate the three new policies and do not establish performance for the expanded index. No additional Nebius calls were made for this update, and no new employee-facing controls or report page were added. A conditional procedural answer may be appropriate for ambiguity only if it acknowledges missing context and avoids unconditional approval; examples are provided for manual review.

## Replit demo deployment

See [REPLIT_SETUP.md](REPLIT_SETUP.md). `python -m scripts.start_app` rebuilds the fictional
search index once per server startup, then launches Streamlit. `.replit` configures
port 8501 and dependency installation. Keep credentials in Replit Secrets.

Rebuild the upload ZIP after code changes with `python -m scripts.package_replit`.
