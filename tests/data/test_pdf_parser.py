"""Generate minimal PDFs at test time; no OCR, GPU or binary fixtures."""

import pymupdf
import pytest

from src.data.ingestion import _parse_records, ingest_directory
from src.data.models import Status


def test_text_pdf(corpus, options):
    path = corpus / "text.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), "First page text")
        doc.new_page().insert_text((72, 72), "Second page text")
        doc.save(path)
    record = list(_parse_records(path, corpus, options))[0]
    assert "First page" in record.text and "Second page" in record.text
    assert record.metadata["page_count"] == 2


def test_blank_and_image_only_pdf(corpus, options):
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(corpus / "blank.pdf")
    with pymupdf.open() as doc:
        page = doc.new_page()
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 8, 8), False)
        pixmap.clear_with(128)
        page.insert_image(pymupdf.Rect(10, 10, 50, 50), pixmap=pixmap)
        doc.save(corpus / "scan.pdf")
    result = ingest_directory(corpus, options)
    assert result.summary["no_extractable_text"] == 2


def test_encrypted_pdf(corpus, options):
    path = corpus / "secret.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), "Protected")
        doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="reader")
    assert list(_parse_records(path, corpus, options))[0].status == Status.PASSWORD_PROTECTED


@pytest.mark.parametrize(
    ("content", "status"), [(b"broken", Status.PARSE_ERROR), (b"", Status.EMPTY_FILE)]
)
def test_invalid_pdf(corpus, options, content, status):
    path = corpus / "broken.pdf"
    path.write_bytes(content)
    assert list(_parse_records(path, corpus, options))[0].status == status
