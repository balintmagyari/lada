"""Routes for the Rheometer workspace: read uploaded exports, and describe the quantities.

The server keeps nothing: `parse` reads the files into JSON and forgets them, and the browser
holds the datasets from then on. `rederive` takes a block back after the user has corrected
what a column is, and returns it with its derived columns and test type recomputed.
"""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile
from pydantic import BaseModel

from ..config import Settings, get_settings
from ..rheometry import quantities as q
from ..rheometry.derive import derive
from ..rheometry.detect import block_temperature, detect_test_type
from ..rheometry.model import FORMAT_LABELS, TEST_TYPE_LABELS, Block, ParseError, ParseResponse
from ..rheometry.readers import ACCEPTED, Limits, read_file
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
        except ValueError as exc:
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
            unit = q.parse_unit(unit_text, override.quantity) if unit_text else spec.canonical
            if unit is None:
                choices = ", ".join(u.symbol or "no unit" for u in spec.units)
                raise ValueError(
                    f"“{unit_text}” is not a unit of {spec.name.lower()}. Choose one of: {choices}."
                )
            column.quantity = override.quantity
            column.unit = spec.canonical.symbol
            column.values = q.to_canonical(raw, unit)
        column.source_unit = unit_text or ""
        column.is_index = False
    derive(block)
    block.test_type = detect_test_type(block)
    block.temperature = block_temperature(block, block.temperature)
    return block
