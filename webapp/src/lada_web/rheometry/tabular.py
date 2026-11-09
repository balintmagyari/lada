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


# Thousands grouped: "1,234.5" and "12,500" in a dot file, "1.234,5" and "1.234" in a comma one.
# The first group never starts with 0, so "0,125" stays a decimal.
_GROUPED_DOT = re.compile(r"^[+-]?[1-9]\d{0,2}(,\d{3})+(\.\d*)?$")
_GROUPED_COMMA = re.compile(r"^[+-]?[1-9]\d{0,2}(\.\d{3})+(,\d*)?$")
# "1,234" or "1.234": a decimal or a thousands group, which only the rest of the file decides.
_AMBIGUOUS = re.compile(r"^[+-]?[1-9]\d{0,2}[.,]\d{3}$")
# A duration written as a clock, h:mm:ss with optional fractional seconds.
_CLOCK = re.compile(r"^(\d+):([0-5]\d):([0-5]\d(?:[.,]\d+)?)$")


def parse_number(cell: str, decimal_comma: bool = False) -> float | None:
    text = cell.strip().replace(" ", "").replace(" ", "")
    if text.lower() in _MISSING:
        return None
    clock = _CLOCK.match(text)
    if clock:
        hours, minutes, seconds = clock.groups()
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds.replace(",", "."))
    if "," in text and "." in text:
        # Both marks: the last one is the decimal mark, the other one groups thousands.
        if _GROUPED_DOT.match(text):
            text = text.replace(",", "")
        elif _GROUPED_COMMA.match(text):
            text = text.replace(".", "").replace(",", ".")
        else:
            return None
    elif "," in text:
        if not decimal_comma and _GROUPED_DOT.match(text):
            text = text.replace(",", "")
        elif text.count(",") == 1:
            # A decimal comma; in a file not flagged as decimal-comma "1,5" still means 1.5.
            text = text.replace(",", ".")
    elif decimal_comma and _GROUPED_COMMA.match(text):
        text = text.replace(".", "")
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
            if _AMBIGUOUS.match(text):
                continue
            if "," in text and "." in text:
                if _GROUPED_DOT.match(text):
                    dot += 1
                elif _GROUPED_COMMA.match(text):
                    comma += 1
            elif re.match(r"^[+-]?\d+,\d+([eE][+-]?\d+)?$", text):
                comma += 1
            elif re.match(r"^[+-]?\d+\.\d+([eE][+-]?\d+)?$", text):
                dot += 1
    return comma > dot


def sniff_delimiter(text: str) -> str | None:
    """The cell delimiter of a text file; None means runs of whitespace."""
    lines = [
        line
        # The first 300 lines are enough, and splitting only the head keeps a huge file cheap.
        for line in text[:300_000].splitlines()[:300]
        if line.strip() and not line.lstrip().startswith(("#", "%", "//", "!"))
    ]
    # Tried in this order and the first that separates cells on most lines wins. A tab or a
    # semicolon is never part of a number, but a comma may be a decimal comma ("0,1;10,5") or
    # a thousands separator ("1\t1,234.5"), so it is chosen only when nothing else fits.
    for delim in ("\t", ";", ","):
        hits = [line.count(delim) for line in lines if delim in line]
        # A delimiter separates cells on most lines, not in a stray sentence here and there.
        if len(hits) < max(2, len(lines) // 2):
            continue
        if delim == ",":
            # "1,5 2,5 3,5": decimal commas between whitespace, not a comma-separated file.
            sample = lines[-20:]
            spaced = sum(bool(re.search(r"\d,\d+\s+[+-]?\d", line)) for line in sample)
            if spaced > len(sample) / 2:
                return None
        return delim
    return None


def too_many_points(limit: int, total: int | None = None) -> ValueError:
    held = f"{total:,} data points, more than" if total is not None else "more than"
    return ValueError(
        f"This file holds {held} the {limit:,} data points this server reads at once. "
        "Export a shorter range or fewer steps."
    )


_WIDE_GAP = re.compile(r"\s{2,}|\t")


def _split_whitespace(line: str) -> list[str]:
    stripped = line.strip()
    tokens = stripped.split()
    # "Shear rate (1/s)  Viscosity (Pa.s)": names with spaces in them, set two or more spaces
    # apart. Split such a header line on the wide gaps only, so each name stays one cell.
    if re.search(r"\S \S", stripped) and _WIDE_GAP.search(stripped):
        numbers = sum(parse_number(t) is not None or t.lower() in _MISSING for t in tokens)
        if numbers < len(tokens) / 2:
            return _WIDE_GAP.split(stripped)
    return tokens


def starts_with_number(row: list[str]) -> bool:
    for cell in row:
        if cell.strip():
            return cell.lstrip()[:1] in "0123456789+-.,"
    return False


def split_rows(text: str, max_rows: int | None = None) -> Rows:
    """The file's rows of cells. With `max_rows`, stop with an error as soon as more rows
    of numbers than that have been read, rather than after splitting a huge file."""
    # Old Mac exports end lines with a bare CR, which the csv module refuses.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    delimiter = sniff_delimiter(text)
    if delimiter is None:
        source = (_split_whitespace(line) for line in io.StringIO(text))
    else:
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        source = ([cell.strip() for cell in row] for row in reader)
    rows: Rows = []
    numeric = 0
    try:
        for row in source:
            rows.append(row)
            if max_rows is not None and starts_with_number(row):
                numeric += 1
                if numeric > max_rows:
                    raise too_many_points(max_rows)
    except csv.Error as exc:
        raise ValueError(
            f"Line {len(rows) + 1} of this file could not be split into cells ({exc}). "
            "Check that it is a plain table exported from the instrument software."
        ) from exc
    return rows


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


def _width(row: list[str]) -> int:
    return sum(1 for c in row if c.strip())


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
        j, data = i, []
        while j < n:
            if _numeric_row(rows[j], decimal_comma):
                data.append(rows[j])
                j += 1
            elif (
                # One blank line inside a table (a page break in the export) does not end it
                # when the rows after it carry on with as many cells.
                _blank(rows[j])
                and j + 1 < n
                and _numeric_row(rows[j + 1], decimal_comma)
                and _width(rows[j + 1]) == _width(data[-1])
            ):
                j += 1
            else:
                break
        if len(data) >= min_rows:
            header, units, start = _header_above(rows, i, decimal_comma)
            tables.append(Table(header, units, data, start, j))
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
    if above >= 0 and _text_row(rows[above], decimal_comma) and _units_row(candidate, rows[above]):
        return rows[above], candidate, above
    return candidate, None, k


# Units of things the registry does not hold (gap, speed, mass...) that still mark a units row.
_OTHER_UNITS = frozenset(
    {
        "mm", "µm", "um", "cm", "m", "rpm", "1/min", "min-1", "mrad", "µrad", "g", "mg",
        "kg", "K/min", "°C/min", "mm/s", "µm/s", "m/s", "bar", "mbar", "V", "A", "W", "J",
    }
)  # fmt: skip
_UNITISH = re.compile(r"^[\w°µμ%/.·⋅^*²³⁻¹\-]{1,12}$")
_UNIT_MARK = re.compile(r"[/%°·⋅.^µμ\d²³⁻¹]")


def _units_row(row: list[str], above: list[str] | None = None) -> bool:
    """A row of units: most cells read as a unit (a known one, a bracketed one, or a short
    unit-like token such as "mrad"), and none of them names a quantity, as a header would."""
    cells = [c.strip() for c in row if c.strip()]
    if not cells or any(q.recognise(c) for c in cells):
        return False
    # A token the registry does not know ("mrad", "µN.m") still reads as a unit when it is
    # short and carries a unit's marks; a plain word ("Gap", "a") does not.
    unitish = sum(
        q.looks_like_unit(c)
        or c in _OTHER_UNITS
        or bool(_UNITISH.match(c) and _UNIT_MARK.search(c))
        for c in cells
    )
    # Under a header that names rheology quantities, half the cells reading as units is enough.
    named = above is not None and any(q.recognise(c) for c in above if c.strip())
    return unitish >= math.ceil((0.5 if named else 0.7) * len(cells))


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
        unit_text = row_unit or header_unit or None
        quantity = q.recognise(label)
        unit = q.parse_unit(unit_text, quantity) if quantity and unit_text else None
        sibling = q.SIBLINGS.get(quantity or "")
        if unit is None and unit_text and sibling and q.parse_unit(unit_text, sibling):
            # "Frequency (rad/s)" is an angular frequency: the unit says which one is meant.
            quantity, unit = sibling, q.parse_unit(unit_text, sibling)
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
