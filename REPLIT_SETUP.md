# Replit fictional-data demo

1. Extract `dist/legal_operations_replit.zip` and upload its contents into a Python
   Replit project. Include hidden `.replit` and `.streamlit/config.toml` files.
   Do not upload your local `.env` or `.venv`.
2. Select Python 3.14 if available (the local tested runtime). Other supported
   versions need a clean install and tests on Replit before publishing. This
   package has not been tested on Replit/Linux yet. Direct dependencies are pinned;
   transitive dependencies are not fully locked.
3. In Tools > Secrets, set `NEBIUS_API_KEY`. Copy non-sensitive configuration
   values from `.env.example` into Replit configurations, including provider,
   endpoint and model names. Do not paste a local `.env` into a public repository.
   Ensure the published deployment receives the same configuration.
4. In the Shell, run:

   ```bash
   python -m pip install -r requirements.txt
   python -m scripts.prepare_app
   python -m unittest discover -s tests -t . -v
   ```

5. Click Run (or run `python -m scripts.start_app`). Startup rebuilds the index before
   starting Streamlit on port 8501. Initial downloads require outbound internet
   and may take longer. Index preparation does not call Nebius. Failure stops
   startup, rather than serving an unprepared app. Do not also launch a second
   server on the same port.
6. Open Preview. Preparation is per server startup, never per chat or Streamlit
   rerun. The small bundled dataset makes the index reconstructible after a
   restart; runtime files and chat sessions are not durable storage.
7. Publish as a single-instance backend deployment; Reserved VM is an option
   for an always-on demo. Confirm the build/run commands from `.replit` and
   expose port 8501. Do not select Static hosting. Restrict demo access before
   sharing, since chat questions spend your Nebius allowance.
8. Verify a fresh restart and the portal. Live questions and live evaluation
   make paid provider calls; run these only when intended. Local checks do not
   establish deployment correctness or the 95% faithfulness target.

The upload bundle includes fictional source files, evaluation scripts and the
original fictional 20-query report JSON required by grouped-report regression
checks. That historical report is not a measurement of the new deployment. It
excludes credentials, installed libraries and the existing Chroma index.

Keep the existing data exclusions in `.gitignore`: this bundle transfers the
explicitly selected fictional files without opening future private files to Git.
For real employee documents, authentication, document permissions, approved
hosted processing and durable operational logging require separate work.

After local changes, regenerate the upload ZIP with `python -m scripts.package_replit`.
Run these module commands from the project root; the package directories must stay intact.
