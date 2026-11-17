"""A written report as a Word document (.docx), to edit or sign it:
black text only, no colours.

Only the Markdown the model is told to use: titles (``#``), paragraphs,
lists (``-``, ``1.``), tables (``|``), bold (``**``) and italics
(``*``). Anything else stays as plain text.
"""
from io import BytesIO
import re

INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*)")
TABLE_RULE = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")


def _runs(paragraph: object, text: str) -> None:
    """Add ``text`` to a paragraph, with its bold and italic parts
    (``code`` marks are dropped)."""
    for piece in INLINE.split(text.replace("`", "")):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            paragraph.add_run(piece[2:-2]).bold = True  # type: ignore
        elif piece.startswith("*") and piece.endswith("*"):
            paragraph.add_run(piece[1:-1]).italic = True  # type: ignore
        else:
            paragraph.add_run(piece)  # type: ignore[attr-defined]


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def to_docx(markdown: str) -> bytes:
    """The document's bytes.

    Args:
        markdown: The report.

    Returns:
        A .docx file.
    """
    from docx import Document
    from docx.shared import Pt, RGBColor

    document = Document()
    style = document.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    # Word's own headings are blue: all in black, for a plain document.
    for name in ("Title", "Heading 1", "Heading 2", "Heading 3"):
        heading = document.styles[name]
        heading.font.color.rgb = RGBColor(0, 0, 0)
        heading.font.name = "Calibri"
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            document.add_heading(heading.group(2).replace("**", ""),
                                 level=min(len(heading.group(1)) - 1, 3)
                                 if len(heading.group(1)) > 1 else 0)
            index += 1
            continue
        if (stripped.startswith("|") and index + 1 < len(lines)
                and TABLE_RULE.match(lines[index + 1].strip())):
            header = _cells(stripped)
            rows = []
            index += 2
            while index < len(lines) and lines[index].strip().startswith(
                    "|"):
                rows.append(_cells(lines[index]))
                index += 1
            table = document.add_table(rows=1 + len(rows),
                                       cols=len(header))
            table.style = "Table Grid"
            for row_number, row in enumerate([header] + rows):
                for column in range(len(header)):
                    cell = table.cell(row_number, column)
                    text = row[column] if column < len(row) else ""
                    paragraph = cell.paragraphs[0]
                    _runs(paragraph, f"**{text}**" if row_number == 0
                          and text else text)
            continue
        bullet = re.match(r"^[-*]\s+(.*)$", stripped)
        numbered = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if bullet:
            _runs(document.add_paragraph(style="List Bullet"),
                  bullet.group(1))
        elif numbered:
            _runs(document.add_paragraph(style="List Number"),
                  numbered.group(1))
        else:
            _runs(document.add_paragraph(), stripped)
        index += 1
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()
