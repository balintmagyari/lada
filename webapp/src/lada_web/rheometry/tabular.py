"""Find tables of numbers in rows of cells, whatever wrote them.

Every reader ends here. A format-specific reader only decides where one block of a file starts
and ends and what metadata surrounds it; this module then finds, inside those rows, the run of
numeric rows, the header above it and an optional units row between the two. Because nothing
here depends on an exact layout, a file whose surrounding layout differs from what a reader
expected still imports, with less metadata at worst.
"""

import csv
import io
import math
import re
from dataclasses import dataclass

from . import quantities as q
from .model import Column

Rows = list[list[str]]

_NUMBER = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_MISSING = frozenset({"", "nan", "-nan", "na", "n/a", "--", "-", "—", "–", "inf", "-inf", "+inf"})
MIN_ROWS = 2


def parse_number(cell: str, decimal_comma: bool = False) -> float | None:
    text = cell.strip().replace(" ", "").replace(" ", "")
    if text.lower() in _MISSING:
        return None
    if decimal_comma:
        if "," in text:
            text = text.replace(".", "").replace(",", ".")
    elif "," in text and "." not in text and text.count(",") == 1:
        # A lone comma in a file not flagged as decimal-comma: "1,5" still means 1.5.
        text = text.replace(",", ".")
    if not _NUMBER.match(text):
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def detect_decimal_comma(rows: Rows) -> bool:
    """True when numbers are written 1,23 (which implies the delimiter is not a comma)."""
    comma = dot = 0
    for row in rows[:400]:
        for cell in row:
            text = cell.strip()
            if re.match(r"^[+-]?\d+,\d+([eE][+-]?\d+)?$", text):
                comma += 1
            elif re.match(r"^[+-]?\d+\.\d+([eE][+-]?\d+)?$", text):
                dot += 1
    return comma > dot


def sniff_delimiter(text: str) -> str | None:
    """The cell delimiter of a text file; None means runs of whitespace."""
    lines = [
        line
        for line in text.splitlines()[:300]
        if line.strip() and not line.lstrip().startswith(("#", "%", "//", "!"))
    ]
    best, best_score = None, 0
    for delim in ("\t", ";", ","):
        hits = [line.count(delim) for line in lines if delim in line]
        # A delimiter separates cells on most lines, not in a stray sentence here and there.
        if len(hits) < max(2, len(lines) // 2):
            continue
        # Prefer the delimiter that appears on many lines the same number of times.
        modal = max(set(hits), key=hits.count)
        score = hits.count(modal) * min(modal, 6)
        if score > best_score:
            best, best_score = delim, score
    if best == ",":
        # "1,5 2,5 3,5": decimal commas between whitespace, not a comma-separated file.
        sample = lines[-20:]
        spaced = sum(bool(re.search(r"\d,\d+\s+[+-]?\d", line)) for line in sample)
        if spaced > len(sample) / 2:
            return None
    return best


def split_rows(text: str) -> Rows:
    delimiter = sniff_delimiter(text)
    if delimiter is None:
        return [line.split() for line in text.splitlines()]
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    return [[cell.strip() for cell in row] for row in reader]


def _numeric_row(row: list[str], decimal_comma: bool) -> bool:
    cells = [c for c in row if c.strip()]
    if not cells:
        return False
    numbers = sum(parse_number(c, decimal_comma) is not None for c in cells)
    return numbers >= max(1, math.ceil(0.6 * len(cells))) and (numbers >= 2 or len(cells) == 1)


def _blank(row: list[str]) -> bool:
    return not any(c.strip() for c in row)


def _text_row(row: list[str], decimal_comma: bool) -> bool:
    return not _blank(row) and not _numeric_row(row, decimal_comma)


@dataclass
class Table:
    header: list[str]
    units: list[str] | None
    data: Rows
    start: int  # index of the header row (or of the first data row when there is none)
    end: int  # one past the last data row


def find_tables(rows: Rows, decimal_comma: bool, min_rows: int = MIN_ROWS) -> list[Table]:
    """Every run of at least `min_rows` numeric rows, with its header and units rows."""
    tables: list[Table] = []
    i, n = 0, len(rows)
    while i < n:
        if not _numeric_row(rows[i], decimal_comma):
            i += 1
            continue
        j = i
        while j < n and _numeric_row(rows[j], decimal_comma):
            j += 1
        if j - i >= min_rows:
            header, units, start = _header_above(rows, i, decimal_comma)
            tables.append(Table(header, units, rows[i:j], start, j))
        i = j
    return tables


def _header_above(rows: Rows, first_data: int, decimal_comma: bool):
    """The header (and units) rows just above a numeric run, skipping blank lines."""
    k = first_data - 1
    while k >= 0 and _blank(rows[k]):
        k -= 1
    if k < 0 or not _text_row(rows[k], decimal_comma):
        return [], None, first_data
    above = k - 1
    while above >= 0 and _blank(rows[above]):
        above -= 1
    candidate = rows[k]
    if above >= 0 and _text_row(rows[above], decimal_comma) and _units_row(candidate):
        return rows[above], candidate, above
    return candidate, None, k


_UNITISH = re.compile(r"^[\w°µμ%/.·⋅^*²³⁻¹\-]{1,12}$")
_UNIT_MARK = re.compile(r"[/%°·⋅.^µμ\d²³⁻¹]")


def _units_row(row: list[str]) -> bool:
    """A row of units: most cells read as a unit (a known one, a bracketed one, or a short
    unit-like token such as "mrad"), and none of them names a quantity, as a header would."""
    cells = [c.strip() for c in row if c.strip()]
    if not cells or any(q.recognise(c) for c in cells):
        return False
    # A token the registry does not know ("mrad", "µN.m") still reads as a unit when it is
    # short and carries a unit's marks; a plain word ("Gap", "a") does not.
    unitish = sum(
        q.looks_like_unit(c) or bool(_UNITISH.match(c) and _UNIT_MARK.search(c)) for c in cells
    )
    return unitish >= math.ceil(0.7 * len(cells))


def _unique(keys: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for key in keys:
        count = seen.get(key, 0)
        seen[key] = count + 1
        out.append(key if count == 0 else f"{key}_{count + 1}")
    return out


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", q.fold(text)).strip("_")


def build_columns(table: Table, decimal_comma: bool) -> list[Column]:
    """The table's columns: recognised, converted to canonical units, empty ones dropped."""
    width = max(len(r) for r in table.data)
    header = list(table.header) + [""] * width
    units = list(table.units or []) + [""] * width
    keys: list[str] = []
    columns: list[Column] = []
    for c in range(width):
        values = [parse_number(r[c], decimal_comma) if c < len(r) else None for r in table.data]
        if all(v is None for v in values):
            continue
        label = header[c].strip() or f"Column {c + 1}"
        name, header_unit = q.split_header(label)
        row_unit = units[c].strip().strip("[]()").strip()
        unit_text = row_unit or header_unit
        quantity = q.recognise(label)
        unit = q.parse_unit(unit_text, quantity) if quantity else None
        if quantity and unit_text and unit is None:
            # Named like a known quantity but in a unit the registry does not know: keep the
            # column unconverted and unrecognised rather than risk a wrong scale.
            quantity = None
        if quantity:
            unit = unit or q.QUANTITY_BY_KEY[quantity].canonical
            values = q.to_canonical(values, unit)
            unit_symbol = q.QUANTITY_BY_KEY[quantity].canonical.symbol
        else:
            unit_symbol = unit_text or ""
        keys.append(quantity or _slug(name) or f"column_{c + 1}")
        columns.append(
            Column(
                key="",
                label=name or label,
                quantity=quantity,
                unit=unit_symbol,
                source_unit=unit_text or "",
                is_index=quantity is None and q.is_index(label),
                values=values,
            )
        )
    for column, key in zip(columns, _unique(keys), strict=True):
        column.key = key
    return columns
