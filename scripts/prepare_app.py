"""Rebuild the fictional search index before serving; no language-model calls."""
from legal_assistant.knowledge_base import build_index


def main():
    count = build_index()
    print(f"Search ready: {count} passages. No Nebius calls made.")


if __name__ == "__main__":
    main()
