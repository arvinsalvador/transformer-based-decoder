"""Single parser registry with lazy imports so the library never imports Streamlit."""

from importlib import import_module

PARSERS = {".txt": "txt_parser", ".csv": "csv_parser", ".pdf": "pdf_parser", ".docx": "docx_parser"}


def get_parser(extension: str):
    """Return a parser callable for a validated extension."""
    return import_module(f"src.data.parsers.{PARSERS[extension.lower()]}").parse
