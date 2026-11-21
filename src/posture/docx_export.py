"""A written report as a Word document (.docx), to edit or sign it:
an A4 technical report in black, the table headers lightly shaded.

Only the Markdown the model is told to use: titles (``#``), paragraphs,
lists (``-``, ``1.``), tables (``|``), bold (``**``) and italics
(``*``). Anything else stays as plain text.

The tables are the hard part: the report's are wide (up to seven
columns, one of them long sentences). Word gives every column the same
width, which broke words in the middle ("Inmediat-a") and spread a table
over three pages; here each column gets the width its text needs.
"""
from io import BytesIO
import math
import re
from typing import Any

INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*)")
TABLE_RULE = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")
# A4, with 2 cm margins: the width of the text, cm.
PAGE_WIDTH_CM, PAGE_HEIGHT_CM, MARGIN_CM = 21.0, 29.7, 2.0
TEXT_WIDTH_CM = PAGE_WIDTH_CM - 2 * MARGIN_CM
# Font sizes, points.
BODY_PT, TABLE_PT = 10.5, 9.0
# About how wide one character of the table's font is, cm (Calibri 9 pt,
# with the cell's inner margins).
CHAR_CM = 0.17
CELL_MARGIN_CM = 0.25
# The header's shading (light grey).
HEADER_FILL = "E7E6E6"


def _runs(paragraph: Any, text: str) -> None:
    """Add ``text`` to a paragraph, with its bold and italic parts
    (``code`` marks are dropped)."""
    for piece in INLINE.split(text.replace("`", "")):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            paragraph.add_run(piece[2:-2]).bold = True
        elif piece.startswith("*") and piece.endswith("*"):
            paragraph.add_run(piece[1:-1]).italic = True
        else:
            paragraph.add_run(piece)


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _plain(text: str) -> str:
    return text.replace("**", "").replace("`", "")


def column_widths(rows: list[list[str]], total: float = TEXT_WIDTH_CM
                  ) -> list[float]:
    """The width of each column, cm, adding up to ``total``.

    Each column needs at least its longest word (so no word is broken)
    and asks for room by how much text it holds (long sentences get the
    most, without starving the short ones: the square root of the
    longest cell). What is left is shared in that proportion.
    """
    columns = max(len(row) for row in rows)
    floor, want = [], []
    for column in range(columns):
        texts = [_plain(row[column]) for row in rows if column < len(row)]
        word = max((len(w) for t in texts for w in t.split()), default=1)
        floor.append(word * CHAR_CM + CELL_MARGIN_CM)
        want.append(math.sqrt(max((len(t) for t in texts), default=1)))
    if sum(floor) >= total:
        # Too many long words to fit: share by need.
        return [total * f / sum(floor) for f in floor]
    room = total - sum(floor)
    return [f + room * w / sum(want) for f, w in zip(floor, want)]


def _shade(cell: Any, fill: str) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shading)


def _row_flag(row: Any, name: str) -> None:
    """A row property: ``tblHeader`` (repeat on each page) or
    ``cantSplit`` (never split between pages)."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    flag = OxmlElement(f"w:{name}")
    flag.set(qn("w:val"), "true")
    row._tr.get_or_add_trPr().append(flag)


def _fixed(table: Any, widths: list[float]) -> None:
    """Give the table those column widths in its grid and a fixed layout:
    Word and LibreOffice read the grid, not the cells' widths."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm

    properties = table._tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    properties.append(layout)
    for column, width in zip(table.columns, widths):
        column.width = Cm(width)
    for grid, width in zip(table._tbl.tblGrid.findall(qn("w:gridCol")),
                           widths):
        grid.set(qn("w:w"), str(int(Cm(width).twips)))


def _table(document: Any, header: list[str], rows: list[list[str]]) -> None:
    from docx.shared import Cm, Pt

    every = [header] + rows
    widths = column_widths(every)
    table = document.add_table(rows=len(every), cols=len(header))
    table.style = "Table Grid"
    table.autofit = False
    _fixed(table, widths)
    for number, row in enumerate(every):
        table_row = table.rows[number]
        _row_flag(table_row, "cantSplit")
        if number == 0:
            _row_flag(table_row, "tblHeader")
        for column, width in enumerate(widths):
            cell = table_row.cells[column]
            cell.width = Cm(width)
            if number == 0:
                _shade(cell, HEADER_FILL)
            text = row[column] if column < len(row) else ""
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.0
            _runs(paragraph, f"**{_plain(text)}**" if number == 0 and text
                  else text)
            for run in paragraph.runs:
                run.font.size = Pt(TABLE_PT)
    # Room after the table, as after a paragraph.
    document.add_paragraph().paragraph_format.space_after = Pt(2)


def _page_number(section: Any) -> None:
    """``Ergovisor · Página N`` at the foot of every page."""
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt

    paragraph = section.footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run("Ergovisor · Página ").font.size = Pt(9)
    run = paragraph.add_run()
    run.font.size = Pt(9)
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            field = OxmlElement("w:fldChar")
            field.set(qn("w:fldCharType"), kind)
        else:
            field = OxmlElement("w:instrText")
            field.set(qn("xml:space"), "preserve")
            field.text = text
        run._r.append(field)


def _styles(document: Any) -> None:
    from docx.shared import Pt, RGBColor

    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(BODY_PT)
    normal.paragraph_format.space_after = Pt(4)
    normal.paragraph_format.line_spacing = 1.1
    # Word's own headings are blue: all in black, for a plain document.
    for name, size, before in (("Title", 18, 0), ("Heading 1", 14, 14),
                               ("Heading 2", 12, 10),
                               ("Heading 3", 11, 8)):
        heading = document.styles[name]
        heading.font.color.rgb = RGBColor(0, 0, 0)
        heading.font.name = "Calibri"
        heading.font.size = Pt(size)
        heading.font.bold = True
        heading.paragraph_format.space_before = Pt(before)
        heading.paragraph_format.space_after = Pt(4)
        heading.paragraph_format.keep_with_next = True
    for name in ("List Bullet",):
        document.styles[name].font.size = Pt(BODY_PT)


def to_docx(markdown: str) -> bytes:
    """The document's bytes.

    Args:
        markdown: The report.

    Returns:
        A .docx file.
    """
    from docx import Document
    from docx.shared import Cm

    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = (Cm(PAGE_WIDTH_CM),
                                               Cm(PAGE_HEIGHT_CM))
    for side in ("left_margin", "right_margin", "top_margin",
                 "bottom_margin"):
        setattr(section, side, Cm(MARGIN_CM))
    _styles(document)
    _page_number(section)
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
            document.add_heading(_plain(heading.group(2)),
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
            _table(document, header, rows)
            continue
        bullet = re.match(r"^[-*]\s+(.*)$", stripped)
        numbered = re.match(r"^(\d+)[.)]\s+(.*)$", stripped)
        if bullet:
            _runs(document.add_paragraph(style="List Bullet"),
                  bullet.group(1))
        elif numbered:
            # The model's own number: Word's numbered style counts on
            # through the whole document, every list after the first
            # would not start at 1.
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.left_indent = Cm(0.63)
            paragraph.paragraph_format.first_line_indent = Cm(-0.63)
            _runs(paragraph, f"{numbered.group(1)}. {numbered.group(2)}")
        else:
            _runs(document.add_paragraph(), stripped)
        index += 1
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()
