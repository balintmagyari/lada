"""Routes that read a LAMMPS output file and return it as a table."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile

from ..config import Settings, get_settings
from ..rendering import table_response
from ..schemas.parsers import DataSection
from ..services import parsers_service
from ..services.export_service import Format, media_type, table_result_from_df, write_table
from ..uploads import download_workspace, file_download, get_workspace, save_upload

router = APIRouter(prefix="/api/parsers", tags=["parsers"])

SettingsDep = Annotated[Settings, Depends(get_settings)]
WorkspaceDep = Annotated[Path, Depends(get_workspace)]
FileDep = Annotated[UploadFile, File()]
FormatQuery = Annotated[Format, Query()]
DataSectionDep = Annotated[DataSection, Depends()]


@router.post("/log/table")
def log_table(request: Request, file: FileDep, settings: SettingsDep, ws: WorkspaceDep):
    path = save_upload(file, ws, settings.max_upload_bytes)
    df = parsers_service.log_table(path)
    result = table_result_from_df(
        df,
        title="Thermodynamic output",
        download_stem=f"{path.stem}_thermo",
        max_rows=settings.max_table_rows,
        meta={"file": path.name},
    )
    return table_response(request, result, download_url="/api/parsers/log/table/download")


@router.post("/log/table/download")
def log_table_download(file: FileDep, settings: SettingsDep, fmt: FormatQuery = "csv"):
    with download_workspace(settings) as ws:
        path = save_upload(file, ws, settings.max_upload_bytes)
        df = parsers_service.log_table(path)
        out = write_table(df, ws, f"{path.stem}_thermo", fmt)
        return file_download(out, ws, media_type(fmt))


@router.post("/dump/preview")
def dump_preview(request: Request, file: FileDep, settings: SettingsDep, ws: WorkspaceDep):
    path = save_upload(file, ws, settings.max_upload_bytes)
    df = parsers_service.dump_preview(path)
    result = table_result_from_df(
        df,
        title="Dump trajectory preview",
        download_stem=f"{path.stem}_preview",
        max_rows=settings.max_table_rows,
        meta={"file": path.name, "frames": len(df)},
    )
    return table_response(request, result, download_url=None)


@router.post("/dump/table")
def dump_table(request: Request, file: FileDep, settings: SettingsDep, ws: WorkspaceDep):
    path = save_upload(file, ws, settings.max_upload_bytes)
    df = parsers_service.dump_table(path)
    result = table_result_from_df(
        df,
        title="Dump trajectory atom table",
        download_stem=f"{path.stem}_atoms",
        max_rows=settings.max_table_rows,
        meta={"file": path.name},
    )
    return table_response(request, result, download_url="/api/parsers/dump/table/download")


@router.post("/dump/table/download")
def dump_table_download(file: FileDep, settings: SettingsDep, fmt: FormatQuery = "csv"):
    with download_workspace(settings) as ws:
        path = save_upload(file, ws, settings.max_upload_bytes)
        df = parsers_service.dump_table(path)
        out = write_table(df, ws, f"{path.stem}_atoms", fmt)
        return file_download(out, ws, media_type(fmt))


@router.post("/data/table")
def data_table(
    request: Request,
    file: FileDep,
    section: DataSectionDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
):
    path = save_upload(file, ws, settings.max_upload_bytes)
    df = parsers_service.data_table(path, section.section)
    result = table_result_from_df(
        df,
        title="Data file section",
        download_stem=f"{path.stem}_{section.section}",
        max_rows=settings.max_table_rows,
        meta={"file": path.name},
    )
    return table_response(request, result, download_url="/api/parsers/data/table/download")


@router.post("/data/table/download")
def data_table_download(
    file: FileDep, section: DataSectionDep, settings: SettingsDep, fmt: FormatQuery = "csv"
):
    with download_workspace(settings) as ws:
        path = save_upload(file, ws, settings.max_upload_bytes)
        df = parsers_service.data_table(path, section.section)
        out = write_table(df, ws, f"{path.stem}_{section.section}", fmt)
        return file_download(out, ws, media_type(fmt))


@router.post("/acf/table")
def acf_table(request: Request, file: FileDep, settings: SettingsDep, ws: WorkspaceDep):
    path = save_upload(file, ws, settings.max_upload_bytes)
    df = parsers_service.acf_table(path)
    result = table_result_from_df(
        df,
        title="Stress ACF table",
        download_stem=f"{path.stem}_acf",
        max_rows=settings.max_table_rows,
        meta={"file": path.name},
    )
    return table_response(request, result, download_url="/api/parsers/acf/table/download")


@router.post("/acf/table/download")
def acf_table_download(file: FileDep, settings: SettingsDep, fmt: FormatQuery = "csv"):
    with download_workspace(settings) as ws:
        path = save_upload(file, ws, settings.max_upload_bytes)
        df = parsers_service.acf_table(path)
        out = write_table(df, ws, f"{path.stem}_acf", fmt)
        return file_download(out, ws, media_type(fmt))
