from pathlib import Path
from types import SimpleNamespace

from ingest import clean_text, page_text, paper_title


def line(text, size, y, direction=(1, 0)):
    return {
        "dir": direction,
        "bbox": (70, y, 500, y + size),
        "spans": [{"text": text, "size": size}],
    }


class PageFixture:
    rect = SimpleNamespace(height=800)

    def get_text(self, kind, **kwargs):
        if kind == "dict":
            return {
                "blocks": [
                    {
                        "lines": [
                            line("Rotated repository stamp", 24, 100, (0, -1)),
                            line("Conference header", 10, 25),
                            line("An Actual Paper Title:", 17, 80),
                            line("Its Second Line", 17, 102),
                            line("First Author, Second Author", 10, 140),
                            line("Abstract", 12, 180),
                            line("A body paragraph", 10, 200),
                        ]
                    }
                ]
            }
        assert kind == "text" and kwargs["sort"] is False
        return "Left column paragraph.\nRight column paragraph."


class DocumentFixture:
    metadata = {}

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return PageFixture()


def test_title_from_multiline_heading_ignores_stamp_and_authors():
    assert (
        paper_title(DocumentFixture(), Path("1234.pdf")) == "An Actual Paper Title: Its Second Line"
    )


def test_metadata_title_takes_priority():
    document = DocumentFixture()
    document.metadata = {"title": "Verified metadata title"}
    assert paper_title(document, Path("1234.pdf")) == "Verified metadata title"


def test_native_reading_order_preserves_column_paragraphs():
    assert page_text(PageFixture()) == "Left column paragraph. Right column paragraph."


def test_ligatures_are_normalized():
    assert clean_text("efﬁcient ﬂow") == "efficient flow"
