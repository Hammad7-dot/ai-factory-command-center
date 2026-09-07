from io import BytesIO

from PIL import Image

from factory.uploads import MAX_DOCUMENT_CHARS, extract_documents, load_image


class Upload:
    def __init__(self, name, payload):
        self.name = name
        self.payload = payload

    def getvalue(self):
        return self.payload


def test_text_upload_is_bounded():
    documents, warnings = extract_documents([Upload("manual.txt", b"a" * (MAX_DOCUMENT_CHARS + 10))])
    assert len(documents[0]["text"]) == MAX_DOCUMENT_CHARS
    assert "truncated" in warnings[0]


def test_oversized_image_dimensions_are_rejected():
    buffer = BytesIO()
    Image.new("RGB", (5000, 4000)).save(buffer, format="PNG")
    buffer.seek(0)
    try:
        load_image(buffer)
    except ValueError as exc:
        assert "at most" in str(exc)
    else:
        raise AssertionError("oversized image was accepted")
