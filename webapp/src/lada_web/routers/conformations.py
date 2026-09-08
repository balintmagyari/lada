"""Routes for chain size and shape calculations from a dump trajectory."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile

from ..config import Settings, get_settings
from ..rendering import table_response
from ..schemas.common import PlotSpec
from ..schemas.conformations import ChainColumns
from ..services import conformations_service
from ..services.export_service import Format, media_type, table_result_from_df, write_table
from ..uploads import download_workspace, file_download, get_workspace, save_upload

router = APIRouter(prefix="/api/conformations", tags=["conformations"])

SettingsDep = Annotated[Settings, Depends(get_settings)]
WorkspaceDep = Annotated[Path, Depends(get_workspace)]
FileDep = Annotated[UploadFile, File()]
FormatQuery = Annotated[Format, Query()]
ColumnsDep = Annotated[ChainColumns, Depends()]


@router.post("/rg")
def rg(
    request: Request,
    file: FileDep,
    columns: ColumnsDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
):
    path = save_upload(file, ws, settings.max_upload_bytes)
    df = conformations_service.rg_sq(
        path,
        coord_cols=columns.coord_cols,
        molecule_col=columns.molecule_col,
        mass_col=columns.mass_col,
    )
    result = table_result_from_df(
        df,
        title="Radius of gyration",
        download_stem=f"{path.stem}_rg_sq",
        max_rows=settings.max_table_rows,
        plot=PlotSpec(x="timestep", y=["rg_sq"], x_label="Timestep", y_label="⟨Rg²⟩"),
        meta={"file": path.name},
    )
    return table_response(request, result, download_url="/api/conformations/rg/download")


@router.post("/rg/download")
def rg_download(
    file: FileDep, columns: ColumnsDep, settings: SettingsDep, fmt: FormatQuery = "csv"
):
    with download_workspace(settings) as ws:
        path = save_upload(file, ws, settings.max_upload_bytes)
        df = conformations_service.rg_sq(
            path,
            coord_cols=columns.coord_cols,
            molecule_col=columns.molecule_col,
            mass_col=columns.mass_col,
        )
        out = write_table(df, ws, f"{path.stem}_rg_sq", fmt)
        return file_download(out, ws, media_type(fmt))


@router.post("/ree")
def ree(
    request: Request,
    file: FileDep,
    columns: ColumnsDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
):
    path = save_upload(file, ws, settings.max_upload_bytes)
    df = conformations_service.ree_sq(
        path,
        coord_cols=columns.coord_cols,
        molecule_col=columns.molecule_col,
        atom_id_col=columns.atom_id_col,
    )
    result = table_result_from_df(
        df,
        title="End-to-end distance",
        download_stem=f"{path.stem}_ree_sq",
        max_rows=settings.max_table_rows,
        plot=PlotSpec(x="timestep", y=["ree_sq"], x_label="Timestep", y_label="⟨Ree²⟩"),
        meta={"file": path.name},
    )
    return table_response(request, result, download_url="/api/conformations/ree/download")


@router.post("/ree/download")
def ree_download(
    file: FileDep, columns: ColumnsDep, settings: SettingsDep, fmt: FormatQuery = "csv"
):
    with download_workspace(settings) as ws:
        path = save_upload(file, ws, settings.max_upload_bytes)
        df = conformations_service.ree_sq(
            path,
            coord_cols=columns.coord_cols,
            molecule_col=columns.molecule_col,
            atom_id_col=columns.atom_id_col,
        )
        out = write_table(df, ws, f"{path.stem}_ree_sq", fmt)
        return file_download(out, ws, media_type(fmt))


@router.post("/ree-vectors")
def ree_vectors(
    request: Request,
    file: FileDep,
    columns: ColumnsDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
):
    path = save_upload(file, ws, settings.max_upload_bytes)
    df = conformations_service.ree_vectors(
        path,
        coord_cols=columns.coord_cols,
        molecule_col=columns.molecule_col,
        atom_id_col=columns.atom_id_col,
    )
    result = table_result_from_df(
        df,
        title="End-to-end vectors",
        download_stem=f"{path.stem}_ree_vectors",
        max_rows=settings.max_table_rows,
        meta={"file": path.name},
    )
    return table_response(request, result, download_url="/api/conformations/ree-vectors/download")


@router.post("/ree-vectors/download")
def ree_vectors_download(
    file: FileDep, columns: ColumnsDep, settings: SettingsDep, fmt: FormatQuery = "csv"
):
    with download_workspace(settings) as ws:
        path = save_upload(file, ws, settings.max_upload_bytes)
        df = conformations_service.ree_vectors(
            path,
            coord_cols=columns.coord_cols,
            molecule_col=columns.molecule_col,
            atom_id_col=columns.atom_id_col,
        )
        out = write_table(df, ws, f"{path.stem}_ree_vectors", fmt)
        return file_download(out, ws, media_type(fmt))
