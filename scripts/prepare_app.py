"""Verify the Pinecone snapshot and prepare the local manifest before serving; no language-model calls."""
from legal_assistant.knowledge_base import build_index


def main():
    count = build_index(upload=False)
    print(f"Search ready: {count} passages. No Nebius calls made.")


if __name__ == "__main__":
    main()
