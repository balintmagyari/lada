"""Routes for chain size and shape calculations from a dump trajectory."""

import threading
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import FileResponse

from ..config import Settings, get_settings
from ..jobs import (
    JobStore,
    get_job_semaphore,
    get_job_store,
    per_file_progress,
    run_job,
    start_workspace,
)
from ..rendering import table_response
from ..schemas.common import FILE_RESPONSES, PlotSpec, StartResponse, TableResult
from ..schemas.conformations import ChainColumns
from ..services import conformations_service, merge
from ..services.export_service import Format, media_type, table_result_from_df, write_table
from ..uploads import download_workspace, file_download, get_workspace, save_uploads

router = APIRouter(prefix="/api/conformations", tags=["conformations"])

SettingsDep = Annotated[Settings, Depends(get_settings)]
WorkspaceDep = Annotated[Path, Depends(get_workspace)]
FilesDep = Annotated[list[UploadFile], File()]
FormatQuery = Annotated[Format, Query()]
ColumnsDep = Annotated[ChainColumns, Depends()]
JobStoreDep = Annotated[JobStore, Depends(get_job_store)]
JobSemaphoreDep = Annotated[threading.Semaphore, Depends(get_job_semaphore)]


def _stem(paths: list[Path]) -> str:
    """The download-stem prefix: a single file's own stem, or an "avgN" label for a repeat set."""
    return paths[0].stem if len(paths) == 1 else f"{paths[0].stem}_avg{len(paths)}"


def _meta(paths: list[Path]) -> dict[str, str | int | float]:
    return {"files": ", ".join(p.name for p in paths), "n_files": len(paths)}


def _rg(
    files: list[UploadFile], ws: Path, settings: Settings, columns: ChainColumns
) -> tuple[list[Path], pd.DataFrame]:
    paths = save_uploads(files, ws, settings.max_upload_bytes, settings.max_files_per_request)
    frames = [
        conformations_service.rg_sq(
            p,
            coord_cols=columns.coord_cols,
            molecule_col=columns.molecule_col,
            mass_col=columns.mass_col,
        )
        for p in paths
    ]
    return paths, merge.average_per_timestep(frames, "rg_sq")


def _ree(
    files: list[UploadFile], ws: Path, settings: Settings, columns: ChainColumns
) -> tuple[list[Path], pd.DataFrame]:
    paths = save_uploads(files, ws, settings.max_upload_bytes, settings.max_files_per_request)
    frames = [
        conformations_service.ree_sq(
            p,
            coord_cols=columns.coord_cols,
            molecule_col=columns.molecule_col,
            atom_id_col=columns.atom_id_col,
        )
        for p in paths
    ]
    return paths, merge.average_per_timestep(frames, "ree_sq")


def _ree_vectors(
    files: list[UploadFile], ws: Path, settings: Settings, columns: ChainColumns
) -> tuple[list[Path], pd.DataFrame]:
    paths = save_uploads(files, ws, settings.max_upload_bytes, settings.max_files_per_request)
    frames = [
        conformations_service.ree_vectors(
            p,
            coord_cols=columns.coord_cols,
            molecule_col=columns.molecule_col,
            atom_id_col=columns.atom_id_col,
        )
        for p in paths
    ]
    return paths, merge.concat_with_repeat_label(frames)


@router.post("/rg", response_model=TableResult)
def rg(
    request: Request,
    files: FilesDep,
    columns: ColumnsDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
):
    paths, df = _rg(files, ws, settings, columns)
    result = table_result_from_df(
        df,
        title="Radius of gyration",
        download_stem=f"{_stem(paths)}_rg_sq",
        max_rows=settings.max_table_rows,
        plot=PlotSpec(x="timestep", y=["rg_sq"], x_label="Timestep", y_label="⟨Rg²⟩"),
        meta=_meta(paths),
    )
    return table_response(request, result, download_url="/api/conformations/rg/download")


@router.post("/rg/download", response_class=FileResponse, responses=FILE_RESPONSES)
def rg_download(
    files: FilesDep, columns: ColumnsDep, settings: SettingsDep, fmt: FormatQuery = "csv"
):
    with download_workspace(settings) as ws:
        paths, df = _rg(files, ws, settings, columns)
        out = write_table(df, ws, f"{_stem(paths)}_rg_sq", fmt)
        return file_download(out, ws, media_type(fmt))


@router.post("/ree", response_model=TableResult)
def ree(
    request: Request,
    files: FilesDep,
    columns: ColumnsDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
):
    paths, df = _ree(files, ws, settings, columns)
    result = table_result_from_df(
        df,
        title="End-to-end distance",
        download_stem=f"{_stem(paths)}_ree_sq",
        max_rows=settings.max_table_rows,
        plot=PlotSpec(x="timestep", y=["ree_sq"], x_label="Timestep", y_label="⟨Ree²⟩"),
        meta=_meta(paths),
    )
    return table_response(request, result, download_url="/api/conformations/ree/download")


@router.post("/ree/download", response_class=FileResponse, responses=FILE_RESPONSES)
def ree_download(
    files: FilesDep, columns: ColumnsDep, settings: SettingsDep, fmt: FormatQuery = "csv"
):
    with download_workspace(settings) as ws:
        paths, df = _ree(files, ws, settings, columns)
        out = write_table(df, ws, f"{_stem(paths)}_ree_sq", fmt)
        return file_download(out, ws, media_type(fmt))


@router.post("/ree-vectors", response_model=TableResult)
def ree_vectors(
    request: Request,
    files: FilesDep,
    columns: ColumnsDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
):
    paths, df = _ree_vectors(files, ws, settings, columns)
    result = table_result_from_df(
        df,
        title="End-to-end vectors",
        download_stem=f"{_stem(paths)}_ree_vectors",
        max_rows=settings.max_table_rows,
        meta=_meta(paths),
    )
    return table_response(request, result, download_url="/api/conformations/ree-vectors/download")


@router.post("/ree-vectors/download", response_class=FileResponse, responses=FILE_RESPONSES)
def ree_vectors_download(
    files: FilesDep, columns: ColumnsDep, settings: SettingsDep, fmt: FormatQuery = "csv"
):
    with download_workspace(settings) as ws:
        paths, df = _ree_vectors(files, ws, settings, columns)
        out = write_table(df, ws, f"{_stem(paths)}_ree_vectors", fmt)
        return file_download(out, ws, media_type(fmt))


@router.post("/rg/start", response_model=StartResponse, status_code=202)
def rg_start(
    files: FilesDep,
    columns: ColumnsDep,
    settings: SettingsDep,
    store: JobStoreDep,
    semaphore: JobSemaphoreDep,
) -> StartResponse:
    ws, paths = start_workspace(
        settings, files, settings.max_upload_bytes, settings.max_files_per_request
    )

    def target(progress_callback):
        frames = [
            conformations_service.rg_sq(
                p,
                coord_cols=columns.coord_cols,
                molecule_col=columns.molecule_col,
                mass_col=columns.mass_col,
                progress_callback=per_file_progress(progress_callback, i, len(paths)),
            )
            for i, p in enumerate(paths)
        ]
        df = merge.average_per_timestep(frames, "rg_sq")
        return table_result_from_df(
            df,
            title="Radius of gyration",
            download_stem=f"{_stem(paths)}_rg_sq",
            max_rows=settings.max_table_rows,
            plot=PlotSpec(x="timestep", y=["rg_sq"], x_label="Timestep", y_label="⟨Rg²⟩"),
            meta=_meta(paths),
        )

    job = run_job(store, ws, target, semaphore, settings.max_concurrent_jobs)
    return StartResponse(job_id=job.id)


@router.post("/ree/start", response_model=StartResponse, status_code=202)
def ree_start(
    files: FilesDep,
    columns: ColumnsDep,
    settings: SettingsDep,
    store: JobStoreDep,
    semaphore: JobSemaphoreDep,
) -> StartResponse:
    ws, paths = start_workspace(
        settings, files, settings.max_upload_bytes, settings.max_files_per_request
    )

    def target(progress_callback):
        frames = [
            conformations_service.ree_sq(
                p,
                coord_cols=columns.coord_cols,
                molecule_col=columns.molecule_col,
                atom_id_col=columns.atom_id_col,
                progress_callback=per_file_progress(progress_callback, i, len(paths)),
            )
            for i, p in enumerate(paths)
        ]
        df = merge.average_per_timestep(frames, "ree_sq")
        return table_result_from_df(
            df,
            title="End-to-end distance",
            download_stem=f"{_stem(paths)}_ree_sq",
            max_rows=settings.max_table_rows,
            plot=PlotSpec(x="timestep", y=["ree_sq"], x_label="Timestep", y_label="⟨Ree²⟩"),
            meta=_meta(paths),
        )

    job = run_job(store, ws, target, semaphore, settings.max_concurrent_jobs)
    return StartResponse(job_id=job.id)


@router.post("/ree-vectors/start", response_model=StartResponse, status_code=202)
def ree_vectors_start(
    files: FilesDep,
    columns: ColumnsDep,
    settings: SettingsDep,
    store: JobStoreDep,
    semaphore: JobSemaphoreDep,
) -> StartResponse:
    ws, paths = start_workspace(
        settings, files, settings.max_upload_bytes, settings.max_files_per_request
    )

    def target(progress_callback):
        frames = [
            conformations_service.ree_vectors(
                p,
                coord_cols=columns.coord_cols,
                molecule_col=columns.molecule_col,
                atom_id_col=columns.atom_id_col,
                progress_callback=per_file_progress(progress_callback, i, len(paths)),
            )
            for i, p in enumerate(paths)
        ]
        df = merge.concat_with_repeat_label(frames)
        return table_result_from_df(
            df,
            title="End-to-end vectors",
            download_stem=f"{_stem(paths)}_ree_vectors",
            max_rows=settings.max_table_rows,
            meta=_meta(paths),
        )

    job = run_job(store, ws, target, semaphore, settings.max_concurrent_jobs)
    return StartResponse(job_id=job.id)
