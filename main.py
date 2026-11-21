"""Diagnostic entry point for the Academic Research Assistant."""

from project.loader import load_documents
from project.config import get_provider


def main():
    count = len(load_documents())
    print(f"Loaded academic papers: {count}")
    print(f"Configured embedding provider: {get_provider()}")


if __name__ == "__main__":
    main()