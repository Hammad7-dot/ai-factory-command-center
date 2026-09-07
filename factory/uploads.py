"""Bound and normalize untrusted files supplied through the dashboard."""
from __future__ import annotations

from io import BytesIO

from PIL import Image
from pypdf import PdfReader

MAX_DOCUMENT_CHARS = 200_000
MAX_PDF_PAGES = 50
MAX_IMAGE_PIXELS = 16_000_000


def extract_documents(files) -> tuple[list[dict], list[str]]:
    """Extract bounded text passages and return user-facing extraction warnings."""
    documents, warnings = [], []
    remaining = MAX_DOCUMENT_CHARS
    for uploaded in files or []:
        if remaining <= 0:
            warnings.append("Document text limit reached; remaining files were skipped.")
            break
        name = str(getattr(uploaded, "name", "uploaded document"))
        try:
            raw = uploaded.getvalue()
            if name.lower().endswith(".pdf"):
                reader = PdfReader(BytesIO(raw))
                pages = reader.pages[:MAX_PDF_PAGES]
                if len(reader.pages) > MAX_PDF_PAGES:
                    warnings.append(f"{name}: only the first {MAX_PDF_PAGES} pages were read.")
                for index, page in enumerate(pages, 1):
                    text = (page.extract_text() or "")[:remaining]
                    if text.strip():
                        documents.append({"source": f"{name} / page {index}", "text": text})
                        remaining -= len(text)
                    if remaining <= 0:
                        warnings.append("Document text limit reached; later content was skipped.")
                        break
            else:
                text = raw.decode("utf-8")[:remaining]
                if text.strip():
                    documents.append({"source": name, "text": text})
                    remaining -= len(text)
                if len(raw) > len(text.encode("utf-8")):
                    warnings.append(f"{name}: text was truncated at the document extraction limit.")
        except Exception as exc:
            warnings.append(f"Unable to extract {name}: {exc}")
    return documents, warnings


def load_image(source) -> Image.Image:
    """Decode an image only when its dimensions are reasonable for this demo."""
    image = Image.open(source)
    width, height = image.size
    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"Image dimensions must contain at most {MAX_IMAGE_PIXELS:,} pixels.")
    image.thumbnail((2048, 2048))
    return image.convert("RGB")
