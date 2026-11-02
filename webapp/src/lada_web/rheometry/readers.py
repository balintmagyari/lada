"""Turn an uploaded file into a Dataset.

    read_file(path, filename, limits) -> Dataset

The format is recognised from the content, not trusted from the extension (a TRIOS export is
often saved as .txt, a RheoCompass export as .csv). Spreadsheets are read sheet by sheet and
each sheet goes through the same recognition as a text file, so a TRIOS or RheoCompass export
pasted into Excel still splits into its steps or intervals.

The TRIOS and RheoCompass splitters only decide where each block starts and which
`key: value` lines describe it. Inside each block, `tabular.find_tables` finds the header,
units row and numbers on its own, so a layout that differs from the one assumed here still
imports.
"""

import hashlib
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from . import tabular
from .derive import derive
from .detect import block_temperature, detect_test_type
from .model import FORMAT_LABELS, Block, Dataset

Rows = tabular.Rows

TEXT_SUFFIXES = {".csv", ".txt", ".dat", ".tsv", ".tab", ".asc", ".prn"}
EXCEL_SUFFIXES = {".xlsx", ".xlsm"}
LEGACY_EXCEL_SUFFIXES = {".xls"}
ACCEPTED = sorted(TEXT_SUFFIXES | EXCEL_SUFFIXES | LEGACY_EXCEL_SUFFIXES)


@dataclass(frozen=True)
class Limits:
    max_rows: int = 200_000
    max_xlsx_uncompressed_bytes: int = 200 * 1024 * 1024


# --- Text --------------------------------------------------------------------------------


def decode(raw: bytes) -> str:
    """Text from bytes in whichever encoding the instrument software used."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    if raw[:200].count(b"\x00") > 20:
        # UTF-16 without a byte-order mark: every other byte of ASCII text is zero.
        return raw.decode("utf-16-le" if raw[1:2] == b"\x00" else "utf-16-be", errors="replace")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


# --- Format recognition ------------------------------------------------------------------


def _first_cell(row: list[str]) -> str:
    for cell in row:
        if cell.strip():
            return cell.strip()
    return ""


def is_trios(rows: Rows) -> bool:
    heads = [_first_cell(r).lower() for r in rows[:2000]]
    if "[step]" in heads:
        return True
    return any(h in ("filename", "instrument serial number", "procedure name") for h in heads[:5])


def is_anton_paar(rows: Rows) -> bool:
    for row in rows[:4000]:
        head = _first_cell(row).lower()
        if head.startswith(("interval and data points", "meas. pts.", "interval data")):
            return True
        if head in ("project:", "test:", "result:"):
            return True
    return False


# --- Metadata ----------------------------------------------------------------------------


def _key_value(row: list[str]) -> tuple[str, str] | None:
    """A `key<TAB>value` or `Key: value` line, as a pair."""
    cells = [c.strip() for c in row if c.strip()]
    # A comment line ("# Sample: PS-1", "% exported by ...") is read for what follows the mark.
    if cells and cells[0][:1] in ("#", "%", ";") or cells and cells[0].startswith("//"):
        rest = " ".join(cells).lstrip("#%;/ ").strip()
        cells = [rest] if rest else []
    if not cells or len(cells) > 3:
        return None
    if len(cells) == 1 and ":" in cells[0]:
        key, _, value = cells[0].partition(":")
        return (key.strip(), value.strip()) if key.strip() and value.strip() else None
    if len(cells) >= 2 and tabular.parse_number(cells[0]) is None:
        return cells[0].rstrip(":").strip(), " ".join(cells[1:])
    return None


def _metadata(rows: Rows) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in rows:
        pair = _key_value(row)
        if pair and pair[0] and pair[0] not in out and len(out) < 60:
            out[pair[0][:80]] = pair[1][:200]
    return out


def _temperature_from(metadata: dict[str, str]) -> float | None:
    for key, value in metadata.items():
        if "temperature" in key.lower() or key.lower() == "temp":
            match = re.search(r"[+-]?\d+(?:[.,]\d+)?", value)
            if match:
                return float(match.group(0).replace(",", "."))
    return None


# --- Splitting a file into segments ------------------------------------------------------


@dataclass
class Segment:
    name: str
    rows: Rows


def _trios_segments(rows: Rows) -> tuple[dict[str, str], list[Segment]]:
    starts = [i for i, r in enumerate(rows) if _first_cell(r).lower() == "[step]"]
    if not starts:
        tables = tabular.find_tables(rows, tabular.detect_decimal_comma(rows))
        first = tables[0].start if tables else len(rows)
        return _metadata(rows[:first]), [Segment("", rows)]
    header = _metadata(rows[: starts[0]])
    segments = []
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(rows)
        body = rows[start + 1 : end]
        name = ""
        for row in body:
            cells = [c.strip() for c in row if c.strip()]
            if cells:
                if len(cells) == 1 and tabular.parse_number(cells[0]) is None:
                    name = cells[0]
                break
        segments.append(Segment(name or f"Step {n + 1}", body))
    return header, segments


def _anton_paar_segments(rows: Rows) -> tuple[dict[str, str], list[Segment]]:
    marks = [
        i
        for i, r in enumerate(rows)
        if _first_cell(r).lower().startswith(("interval and data points", "meas. pts."))
    ]
    if not marks:
        return _metadata(rows[:50]), [Segment("", rows)]
    header = _metadata(rows[: marks[0]])
    result_name = header.get("Result") or header.get("Test") or ""
    segments = []
    for n, start in enumerate(marks):
        end = marks[n + 1] if n + 1 < len(marks) else len(rows)
        # A "Result:" line between intervals names the measurements that follow it.
        for row in rows[marks[n - 1] if n else 0 : start]:
            cells = [c.strip() for c in row if c.strip()]
            if cells and cells[0].lower() == "result:" and len(cells) > 1:
                result_name = cells[1]
        numbers = [v for v in (tabular.parse_number(c) for c in rows[start]) if v is not None]
        interval = f"Interval {int(numbers[0])}" if numbers else f"Interval {n + 1}"
        segments.append(
            Segment(f"{result_name}, {interval}" if result_name else interval, rows[start:end])
        )
    return header, segments


# --- Building a dataset ------------------------------------------------------------------


def _blocks_from(segments: list[Segment], decimal_comma: bool, prefix: str) -> list[Block]:
    blocks: list[Block] = []
    for segment in segments:
        tables = tabular.find_tables(segment.rows, decimal_comma)
        for t_index, table in enumerate(tables):
            columns = tabular.build_columns(table, decimal_comma)
            if not columns:
                continue
            local = _metadata(segment.rows[: table.start])
            name = segment.name
            if len(tables) > 1:
                name = f"{name} ({t_index + 1})" if name else f"Table {t_index + 1}"
            if prefix:
                name = f"{prefix}: {name}" if name else prefix
            name = name or "Table"
            block = Block(
                id="", name=name[:120], n_points=len(table.data), metadata=local, columns=columns
            )
            derive(block)
            block.test_type = detect_test_type(block)
            block.temperature = block_temperature(block, _temperature_from(local))
            blocks.append(block)
    return blocks


def _rows_to_blocks(rows: Rows, prefix: str) -> tuple[str, dict[str, str], list[Block], bool]:
    decimal_comma = tabular.detect_decimal_comma(rows)
    if is_trios(rows):
        fmt = "trios"
        metadata, segments = _trios_segments(rows)
    elif is_anton_paar(rows):
        fmt = "anton_paar"
        metadata, segments = _anton_paar_segments(rows)
    else:
        fmt = "delimited"
        tables = tabular.find_tables(rows, decimal_comma)
        first = tables[0].start if tables else len(rows)
        metadata, segments = _metadata(rows[:first]), [Segment("", rows)]
    return fmt, metadata, _blocks_from(segments, decimal_comma, prefix), decimal_comma


def _cell_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value).strip()


def _xlsx_sheets(path: Path, limits: Limits) -> list[tuple[str, Rows]]:
    try:
        with zipfile.ZipFile(path) as archive:
            unpacked = sum(info.file_size for info in archive.infolist())
    except zipfile.BadZipFile as exc:
        raise ValueError("This file is not a readable .xlsx workbook.") from exc
    if unpacked > limits.max_xlsx_uncompressed_bytes:
        limit = limits.max_xlsx_uncompressed_bytes // (1024 * 1024)
        raise ValueError(
            f"This workbook unpacks to more than {limit} MB, which is more than this server "
            "reads. Save the sheet you need on its own, or export it as CSV."
        )
    import openpyxl

    try:
        book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:  # openpyxl raises many types for a damaged workbook
        raise ValueError("This file is not a readable .xlsx workbook.") from exc
    try:
        sheets: list[tuple[str, Rows]] = []
        cells = 0
        for sheet in book.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                rows.append([_cell_text(v) for v in row])
                cells += len(row)
                if cells > limits.max_rows * 40:
                    raise ValueError(
                        "This workbook holds more cells than this server reads at once. "
                        "Save the sheet you need on its own, or export it as CSV."
                    )
            sheets.append((sheet.title, rows))
        return sheets
    finally:
        book.close()


def _xls_sheets(path: Path) -> list[tuple[str, Rows]]:
    import xlrd

    try:
        book = xlrd.open_workbook(str(path), on_demand=True)
    except Exception as exc:  # xlrd raises XLRDError and others for a damaged file
        raise ValueError("This file is not a readable .xls workbook.") from exc
    try:
        return [
            (sheet.name, [[_cell_text(v) for v in sheet.row_values(r)] for r in range(sheet.nrows)])
            for sheet in book.sheets()
        ]
    finally:
        book.release_resources()


def read_file(path: Path, filename: str, limits: Limits | None = None) -> Dataset:
    limits = limits or Limits()
    suffix = Path(filename).suffix.lower()
    raw = path.read_bytes()

    if suffix in EXCEL_SUFFIXES or raw[:4] == b"PK\x03\x04":
        sheets, spreadsheet = _xlsx_sheets(path, limits), True
    elif suffix in LEGACY_EXCEL_SUFFIXES or raw[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        sheets, spreadsheet = _xls_sheets(path), True
    else:
        sheets, spreadsheet = [("", tabular.split_rows(decode(raw)))], False

    blocks: list[Block] = []
    metadata: dict[str, str] = {}
    formats: list[str] = []
    decimal_comma = False
    for title, rows in sheets:
        fmt, meta, sheet_blocks, comma = _rows_to_blocks(rows, title if len(sheets) > 1 else "")
        if sheet_blocks:
            formats.append(fmt)
        decimal_comma |= comma
        for key, value in meta.items():
            metadata.setdefault(key, value)
        blocks.extend(sheet_blocks)

    if not blocks:
        raise ValueError(
            "No table of numbers was found in this file. A readable file has a header row "
            "naming each column, then at least two rows of numbers."
        )
    total = sum(b.n_points for b in blocks)
    if total > limits.max_rows:
        raise ValueError(
            f"This file holds {total:,} data points, more than the {limits.max_rows:,} this "
            "server reads at once. Export a shorter range or fewer steps."
        )

    warnings: list[str] = []
    instrument = [f for f in formats if f in ("trios", "anton_paar")]
    fmt = "excel" if spreadsheet else (instrument[0] if instrument else "delimited")
    if spreadsheet and instrument:
        label = FORMAT_LABELS[instrument[0]].removesuffix(" export")
        warnings.append(f"This spreadsheet holds a {label} export; it was split the same way.")
    if decimal_comma:
        warnings.append("Numbers were written with decimal commas and were read as such.")
    for block in blocks:
        unknown = [c.label for c in block.columns if c.quantity is None and not c.is_index]
        if unknown:
            names = ", ".join(f"“{u}”" for u in unknown[:6])
            more = f" and {len(unknown) - 6} more" if len(unknown) > 6 else ""
            warnings.append(
                f"{block.name}: {names}{more} not recognised as a rheology quantity. Set it "
                "on the column if you know what it is; Custom plot can draw it either way."
            )

    digest = hashlib.sha1(filename.encode() + raw).hexdigest()[:12]
    for n, block in enumerate(blocks):
        block.id = f"{digest}-{n}"
    return Dataset(
        id=digest,
        filename=filename,
        format=fmt,  # type: ignore[arg-type]
        format_label=FORMAT_LABELS[fmt],
        metadata=metadata,
        warnings=warnings[:30],
        blocks=blocks,
    )
