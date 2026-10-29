"""Create an explicit demo upload bundle without local credentials or caches."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = [ROOT / name for name in (
        "app.py", "requirements.txt", ".replit", ".env.example", ".gitignore",
        "README.md", "REPLIT_SETUP.md", ".streamlit/config.toml",
        "evaluation/scenario_eval_set.json",
        "storage/evaluations/resolution_20_queries/resolution_report.json")]
    for folder in ("legal_assistant", "evaluation", "scripts", "tests"):
        files.extend((ROOT / folder).rglob("*.py"))
    # Enumerate approved fictional filenames rather than all future data files.
    for line in (ROOT / ".gitignore").read_text().splitlines():
        if line.startswith("!data/"):
            path = ROOT / line[1:]
            if path.is_file() and path.suffix in {".json", ".pdf"}:
                files.append(path)
    output = ROOT / "dist/legal_operations_replit.zip"
    output.parent.mkdir(exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.relative_to(ROOT))
    print(f"Created {output} with {len(set(files))} files.")


if __name__ == "__main__":
    main()
