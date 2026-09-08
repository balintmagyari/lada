"""Routes for relaxation modulus and dynamic moduli calculations."""

from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import FileResponse

from ..config import Settings, get_settings
from ..rendering import table_response
from ..schemas.common import FILE_RESPONSES, PlotSpec, TableResult
from ..schemas.rheology import PronyParams, ThermoState
from ..services import rheology_service
from ..services.export_service import Format, media_type, table_result_from_df, write_table
from ..uploads import download_workspace, file_download, get_workspace, save_upload

router = APIRouter(prefix="/api/rheology", tags=["rheology"])

SettingsDep = Annotated[Settings, Depends(get_settings)]
WorkspaceDep = Annotated[Path, Depends(get_workspace)]
FileDep = Annotated[UploadFile, File()]
FormatQuery = Annotated[Format, Query()]
ThermoStateDep = Annotated[ThermoState, Depends()]
PronyParamsDep = Annotated[PronyParams, Depends()]

G_T_PLOT = PlotSpec(x="lag_time", y=["G_GK", "G_FSR"], x_label="Lag time", y_label="G(t)")


def _moduli_plot(df: pd.DataFrame) -> PlotSpec:
    y = [c for c in df.columns if c != "omega"]
    return PlotSpec(x="omega", y=y, x_label="ω", y_label="G′, G″", x_log=True, y_log=True)


@router.post("/stress-relaxation", response_model=TableResult)
def stress_relaxation(
    request: Request,
    file: FileDep,
    state: ThermoStateDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
):
    path = save_upload(file, ws, settings.max_upload_bytes)
    df = rheology_service.stress_relaxation(path, state.volume, state.temperature)
    result = table_result_from_df(
        df,
        title="Stress relaxation modulus",
        download_stem=f"{path.stem}_G_t",
        max_rows=settings.max_table_rows,
        plot=G_T_PLOT,
        meta={"file": path.name},
    )
    return table_response(request, result, download_url="/api/rheology/stress-relaxation/download")


@router.post("/stress-relaxation/download", response_class=FileResponse, responses=FILE_RESPONSES)
def stress_relaxation_download(
    file: FileDep, state: ThermoStateDep, settings: SettingsDep, fmt: FormatQuery = "csv"
):
    with download_workspace(settings) as ws:
        path = save_upload(file, ws, settings.max_upload_bytes)
        df = rheology_service.stress_relaxation(path, state.volume, state.temperature)
        out = write_table(df, ws, f"{path.stem}_G_t", fmt)
        return file_download(out, ws, media_type(fmt))


@router.post("/dynamic-moduli", response_model=TableResult)
def dynamic_moduli(
    request: Request,
    file: FileDep,
    state: ThermoStateDep,
    prony: PronyParamsDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
):
    path = save_upload(file, ws, settings.max_upload_bytes)
    df = rheology_service.dynamic_moduli(
        path,
        state.volume,
        state.temperature,
        method=prony.method,
        t_min=prony.t_min,
        t_cutoff=prony.t_cutoff,
        n_modes=prony.n_modes,
        n_omega=prony.n_omega,
        omega_min=prony.omega_min,
        omega_max=prony.omega_max,
    )
    result = table_result_from_df(
        df,
        title="Dynamic moduli",
        download_stem=f"{path.stem}_moduli",
        max_rows=settings.max_table_rows,
        plot=_moduli_plot(df),
        meta={"file": path.name},
    )
    return table_response(request, result, download_url="/api/rheology/dynamic-moduli/download")


@router.post("/dynamic-moduli/download", response_class=FileResponse, responses=FILE_RESPONSES)
def dynamic_moduli_download(
    file: FileDep,
    state: ThermoStateDep,
    prony: PronyParamsDep,
    settings: SettingsDep,
    fmt: FormatQuery = "csv",
):
    with download_workspace(settings) as ws:
        path = save_upload(file, ws, settings.max_upload_bytes)
        df = rheology_service.dynamic_moduli(
            path,
            state.volume,
            state.temperature,
            method=prony.method,
            t_min=prony.t_min,
            t_cutoff=prony.t_cutoff,
            n_modes=prony.n_modes,
            n_omega=prony.n_omega,
            omega_min=prony.omega_min,
            omega_max=prony.omega_max,
        )
        out = write_table(df, ws, f"{path.stem}_moduli", fmt)
        return file_download(out, ws, media_type(fmt))
