"""Routes for the Rheometer workspace: read uploaded exports, and describe the quantities.

The server keeps nothing: `parse` reads the files into JSON and forgets them, and the browser
holds the datasets from then on. `rederive` takes a block back after the user has corrected
what a column is, and returns it with its derived columns and test type recomputed.
"""

import csv
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from ..config import Settings, get_settings
from ..rheometry import quantities as q
from ..rheometry.derive import derive
from ..rheometry.detect import block_temperature, detect_test_type, metadata_temperature
from ..rheometry.figure import MEDIA_TYPES, FigureSpec, pgfplots_zip, render_with_notes
from ..rheometry.model import FORMAT_LABELS, TEST_TYPE_LABELS, Block, ParseError, ParseResponse
from ..rheometry.readers import ACCEPTED, Limits, read_file
from ..rheometry.tts import TTSRequest, TTSResult, shift_factors
from ..uploads import get_workspace, save_uploads

router = APIRouter(prefix="/api/rheometry", tags=["rheometry"])

SettingsDep = Annotated[Settings, Depends(get_settings)]
WorkspaceDep = Annotated[Path, Depends(get_workspace)]
FilesDep = Annotated[list[UploadFile], File()]


@router.post("/parse", response_model=ParseResponse)
def parse(files: FilesDep, settings: SettingsDep, ws: WorkspaceDep) -> ParseResponse:
    """Read rheometer exports and spreadsheets. A file that cannot be read is reported by name
    in `errors` and does not stop the others."""
    paths = save_uploads(files, ws, settings.max_upload_bytes, settings.max_files_per_request)
    limits = Limits(
        max_rows=settings.max_rheometry_rows,
        max_xlsx_uncompressed_bytes=settings.max_xlsx_uncompressed_bytes,
    )
    response = ParseResponse(datasets=[])
    for upload, path in zip(files, paths, strict=True):
        name = Path(upload.filename or path.name).name
        try:
            response.datasets.append(read_file(path, name, limits))
        except (ValueError, csv.Error) as exc:
            response.errors.append(ParseError(filename=name, message=str(exc)))
    return response


@router.get("/quantities")
def quantities() -> dict:
    """Everything the browser needs to label and convert: quantities with their units, the
    names of formats and test types, and the file types the importer reads."""
    return {
        "quantities": q.registry_json(),
        "formats": FORMAT_LABELS,
        "test_types": TEST_TYPE_LABELS,
        "accepted": ACCEPTED,
    }


class ColumnOverride(BaseModel):
    key: str
    quantity: str | None  # None: not a rheology quantity
    unit: str | None = None  # the unit the file's numbers are in; None keeps the current one


class RederiveRequest(BaseModel):
    block: Block
    overrides: list[ColumnOverride] = []


def _source_unit(column) -> q.Unit | None:
    if column.quantity is None:
        return None
    spec = q.QUANTITY_BY_KEY[column.quantity]
    # An empty source unit means the file gave none and the canonical unit was assumed; it
    # never means the unit spelt "" (a strain read as a fraction records "1" instead).
    if not column.source_unit:
        return spec.canonical
    return q.parse_unit(column.source_unit, column.quantity) or spec.canonical


@router.post("/rederive", response_model=Block)
def rederive(body: RederiveRequest) -> Block:
    """Apply the user's corrections to a block's columns, then recompute what follows."""
    block = body.block
    by_key = {c.key: c for c in block.columns}
    for override in body.overrides:
        column = by_key.get(override.key)
        if column is None or column.derived:
            raise ValueError(f"This block has no measured column “{override.key}”.")
        if override.quantity is not None and override.quantity not in q.QUANTITY_BY_KEY:
            raise ValueError(f"“{override.quantity}” is not a quantity this app knows.")
        # Back to the numbers as the file wrote them...
        old = _source_unit(column)
        raw = (
            [None if v is None else (v - old.offset) / old.factor for v in column.values]
            if old
            else list(column.values)
        )
        unit_text = column.source_unit if override.unit is None else override.unit.strip()
        # ...then forward into the newly chosen quantity.
        if override.quantity is None:
            column.quantity, column.unit, column.values = None, unit_text, raw
        else:
            spec = q.QUANTITY_BY_KEY[override.quantity]
            if unit_text or override.unit is not None:
                # Named by the user, "" included: for a strain that is the fraction.
                unit = q.parse_unit(unit_text, override.quantity)
                if unit is None and not unit_text:
                    unit = spec.canonical
            else:
                unit = spec.canonical
            if unit is None:
                choices = ", ".join(u.symbol or "no unit" for u in spec.units)
                raise ValueError(
                    f"“{unit_text}” is not a unit of {spec.name.lower()}. Choose one of: {choices}."
                )
            column.quantity = override.quantity
            column.unit = spec.canonical.symbol
            column.values = q.to_canonical(raw, unit)
            if not unit_text and unit != spec.canonical:
                unit_text = unit.aliases[0]
        column.source_unit = unit_text or ""
        column.is_index = False
    derive(block)
    block.test_type = detect_test_type(block)
    # From the columns as they are now, else from the block's metadata, as on import: never
    # the old value, which may have come from a column that is no longer a temperature.
    block.temperature = block_temperature(block, metadata_temperature(block.metadata))
    return block


# --- Export ---------------------------------------------------------------------------------

FIGURE_RESPONSES = {
    200: {
        "description": "The figure, as an attachment (or inline for the studio's preview).",
        "content": {
            media: {"schema": {"type": "string", "format": "binary"}}
            for media in MEDIA_TYPES.values()
        },
    }
}


@router.post("/figure", response_class=Response, responses=FIGURE_RESPONSES)
def figure(
    spec: FigureSpec,
    fmt: Annotated[Literal["svg", "pdf", "png"], Query()] = "svg",
    dpi: Annotated[int, Query(ge=72, le=1200)] = 600,
    preview: Annotated[bool, Query()] = False,
) -> Response:
    """The studio's figure drawn for print: the same series, sizes and style the browser shows,
    rendered with matplotlib. `preview=true` returns it inline for the Figure tab."""
    data, notes = render_with_notes(spec, fmt, dpi)
    disposition = "inline" if preview else f'attachment; filename="{spec.filename}.{fmt}"'
    headers = {"Content-Disposition": disposition, "Cache-Control": "no-store"}
    if notes:
        # Advice for the studio to show under the preview; plain ASCII, as headers must be.
        headers["X-Figure-Notes"] = " ".join(notes)
    return Response(content=data, media_type=MEDIA_TYPES[fmt], headers=headers)


@router.post(
    "/pgfplots",
    response_class=Response,
    responses={
        200: {
            "description": "figure.tex and its data tables, zipped.",
            "content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}},
        }
    },
)
def pgfplots(spec: FigureSpec, ws: WorkspaceDep) -> Response:
    """The same figure as a pgfplots source: figure.tex plus one data table per series."""
    data = pgfplots_zip(spec, ws)
    return Response(
        content=data,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{spec.filename}-pgfplots.zip"',
            "Cache-Control": "no-store",
        },
    )


# --- Time-temperature superposition -----------------------------------------------------------


@router.post("/tts", response_model=TTSResult)
def tts(body: TTSRequest) -> TTSResult:
    """Shift factors that lay frequency sweeps at several temperatures onto one master curve
    at the reference temperature, with WLF and Arrhenius fits to them. The browser applies
    the shifts itself, so the master curve is drawn, styled and exported like any graph."""
    return shift_factors(body)
