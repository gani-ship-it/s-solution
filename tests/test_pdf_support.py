"""Tests for PDF document and arXiv whitepaper extraction support."""

import io
from unittest.mock import patch, MagicMock
from pypdf import PdfWriter

from src.tools.fetcher import extract_pdf_content, fetch_webpage


def create_sample_pdf_bytes(title: str, text: str) -> bytes:
    """Helper to generate valid in-memory PDF bytes with text."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    writer.add_metadata({"/Title": title})
    
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_extract_pdf_content_extracts_title_and_handles_pdf_bytes():
    """Verify extract_pdf_content parses PDF bytes and retrieves title."""
    pdf_bytes = create_sample_pdf_bytes(
        title="Scaling Fault-Tolerant Quantum Architectures",
        text="Sample text content inside the academic whitepaper."
    )

    title, content = extract_pdf_content(pdf_bytes, url="https://arxiv.org/pdf/2401.99999.pdf")
    assert title == "Scaling Fault-Tolerant Quantum Architectures"
    assert isinstance(content, str)


def test_fetch_webpage_handles_application_pdf_content_type():
    """Verify fetch_webpage detects PDF content-type and registers source as S1."""
    pdf_bytes = create_sample_pdf_bytes(
        title="Neural Architecture Optimization",
        text="Detailed mathematical proofs and experimental benchmarks."
    )
    fetched_sources = {}

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/pdf"}
        mock_resp.content = pdf_bytes
        mock_get.return_value = mock_resp

        # Mock page text extraction
        with patch("pypdf.PdfReader") as mock_reader_cls:
            mock_reader = MagicMock()
            mock_reader.metadata.title = "Neural Architecture Optimization"
            mock_page = MagicMock()
            mock_page.extract_text.return_value = (
                "Neural Architecture Optimization demonstrates a 40% reduction in training latency. "
                "The mathematical framework ensures convergence under non-convex loss surfaces."
            )
            mock_reader.pages = [mock_page]
            mock_reader_cls.return_value = mock_reader

            source, error = fetch_webpage("https://arxiv.org/pdf/2401.12345.pdf", fetched_sources=fetched_sources)

        assert error is None
        assert source is not None
        assert source["id"] == "S1"
        assert source["title"] == "Neural Architecture Optimization"
        assert "40% reduction" in source["content"]


def test_fetch_webpage_handles_corrupt_pdf_gracefully():
    """Verify corrupt PDF bytes are caught with informative error without crashing."""
    corrupt_bytes = b"not a valid pdf binary stream"
    fetched_sources = {}

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/pdf"}
        mock_resp.content = corrupt_bytes
        mock_get.return_value = mock_resp

        source, error = fetch_webpage("https://example.com/broken.pdf", fetched_sources=fetched_sources)

        assert source is None
        assert "Failed to parse PDF" in error
