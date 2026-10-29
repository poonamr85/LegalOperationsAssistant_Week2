# Replit: Pinecone fictional-data demo

This migration is on `feature/pinecone`; `main` still contains Chroma until a later
review/merge. Import the migrated branch after it has been pushed, or extract the
latest `dist/legal_operations_replit.zip` and upload its contents. Include hidden
`.replit` and `.streamlit/config.toml`. Never upload `.env` or `.venv`.

1. The compatible index `legal-operations-minilm-vectors` is populated with 49
   passages. Use its host from your local `.env`. Required settings are: bring-your-own vectors, 384 dimensions, cosine.
   A 384-dimensional hosted Llama embedding index is not interchangeable with
   MiniLM. The code rejects hosted-embedding schemas. The one-time `scripts.create_pinecone_index` setup script was used to create
   this index; normal app startup never creates or deletes indexes.
2. Set `PINECONE_API_KEY` and `NEBIUS_API_KEY` in Replit Secrets. Set index name,
   host, namespace prefix, provider endpoint and model names from `.env.example`
   in Configurations. Ensure published deployments receive these variables.
3. Install and run local regression checks from the project root:

   ```bash
   python -m scripts.build_environment
   .venv/bin/python -m unittest discover -s tests -t . -v
   ```

   Tests mock Pinecone requests; initial MiniLM downloads need outbound internet.
   Python 3.14 is tested locally. Linux/Replit dependency availability and hosting
   must still be verified; direct versions are pinned, not every transitive package.
4. The initial dataset is already uploaded. As an administrator, upload again
   only when the bundled fictional documents or embedding implementation change:

   ```bash
   .venv/bin/python -m legal_assistant.knowledge_base
   ```

   This incurs Pinecone operations but no Nebius calls. Do not put it in the
   deployment run command. It is safe to rerun for the same snapshot, although
   repeated uploads incur writes. New document snapshots use different namespaces;
   older snapshots remain until explicitly cleaned up.
5. Click Run or use `.venv/bin/python -m scripts.start_app`. Startup verifies the existing
   snapshot and creates the local manifest, then starts Streamlit on port 8501.
   It does not upload/rebuild remote vectors. First model download may be slow.
   Startup fails if the remote snapshot is missing; run Step 4 rather than trying
   to serve an empty index. Do not start a second server on the same port.
6. After Preview loads, configure a single-instance backend deployment. Reserved
   VM is an always-on option with ongoing cost; Static hosting cannot run Python.
   Build: `python -m scripts.build_environment`.
   Run: `.venv/bin/python -m scripts.start_app`. Expose internal port 8501.
7. Confirm restart behavior and restrict demo access. Normal chat and live
   evaluation use paid Nebius calls; Pinecone queries also consume service usage.
   Neither live answers nor Replit hosting were verified during offline tests.

Generate the updated upload ZIP with `python -m scripts.package_replit`. The ZIP
includes fictional samples and one historical evaluation JSON needed by regression
tests, but excludes credentials, indexes and model caches. Saved Word guides and
architecture images still describe the former Chroma version; use README.md for
current implementation details. Keep real employee data out of this prototype.

## Final publishing checklist

The pinned UI pair is Streamlit 1.66.1 and Pillow 13.0.0. Keep the Python-specific
NumPy pins. The build script creates `.venv` if needed, bootstraps pip with
`ensurepip`, installs requirements with `PIP_USER=false` and runs `pip check`.
It preserves index/proxy/firewall configuration; no system packages are changed.
The base Python toolchain must provide `venv` and `ensurepip`. If either is absent,
configure a full Python toolchain in Replit before publishing.

In Publishing > Adjust settings, verify:

- Deployment: single-instance backend (Reserved VM for an always-on demo; ongoing cost).
- Build command: `python -m scripts.build_environment`.
- Run command: `.venv/bin/python -m scripts.start_app`.
- Port: 8501 internally, exposed as the web application.
- Deployment secrets: `NEBIUS_API_KEY`, `PINECONE_API_KEY`.
- Deployment configuration: Pinecone index name/host/namespace, Nebius provider,
  endpoint and model names from your working Preview.
- Access: restrict to the intended demo audience; all chat uses your provider accounts.

Run Preview using `.venv/bin/python -m scripts.start_app` before clicking Publish.
Do not run resource creation or vector uploads as a build/run command. After
publishing, open the generated URL and confirm the Overview, navigation and a
fresh restart. Live answer tests incur Nebius usage and are not automated here.

When pulling changes into a Replit branch with local edits, preserve the edits
first and merge explicitly; do not use a destructive reset. Confirm the pinned
versions and the build/run commands after any merge.
