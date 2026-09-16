"""Routes for relaxation modulus and dynamic moduli calculations from one or more ACF files."""

import threading
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import FileResponse

from ..config import Settings, get_settings
from ..jobs import JobStore, get_job_semaphore, get_job_store, run_job, start_workspace
from ..rendering import table_response
from ..schemas.common import FILE_RESPONSES, PlotSpec, StartResponse, TableResult
from ..schemas.regimes import DynamicModuliFromRegimesRequest, RheologyMergeRequest, RheologyRegime
from ..schemas.rheology import PronyParams, ThermoState
from ..services import merge, rheology_service
from ..services.export_service import Format, media_type, table_result_from_df, write_table
from ..uploads import download_workspace, file_download, get_workspace, save_uploads

router = APIRouter(prefix="/api/rheology", tags=["rheology"])

SettingsDep = Annotated[Settings, Depends(get_settings)]
WorkspaceDep = Annotated[Path, Depends(get_workspace)]
FilesDep = Annotated[list[UploadFile], File()]
FormatQuery = Annotated[Format, Query()]
ThermoStateDep = Annotated[ThermoState, Depends()]
PronyParamsDep = Annotated[PronyParams, Depends()]
JobStoreDep = Annotated[JobStore, Depends(get_job_store)]
JobSemaphoreDep = Annotated[threading.Semaphore, Depends(get_job_semaphore)]

G_T_PLOT = PlotSpec(x="lag_time", y=["G_GK", "G_FSR"], x_label="Lag time", y_label="G(t)")


def _stem(paths: list[Path]) -> str:
    """The download-stem prefix: a single file's own stem, or an "avgN" label for a repeat set."""
    return paths[0].stem if len(paths) == 1 else f"{paths[0].stem}_avg{len(paths)}"


def _meta(paths: list[Path]) -> dict[str, str | int | float]:
    return {"files": ", ".join(p.name for p in paths), "n_files": len(paths)}


def _moduli_plot(df: pd.DataFrame) -> PlotSpec:
    y = [c for c in df.columns if c != "omega"]
    return PlotSpec(x="omega", y=y, x_label="ω", y_label="G′, G″", x_log=True, y_log=True)


def _stress_relaxation(
    files: list[UploadFile], ws: Path, settings: Settings, state: ThermoState
) -> tuple[list[Path], pd.DataFrame]:
    paths = save_uploads(files, ws, settings.max_upload_bytes, settings.max_files_per_request)
    frames = [rheology_service.stress_relaxation(p, state.volume, state.temperature) for p in paths]
    return paths, merge.average_rheology(frames, ("G_GK", "G_FSR"))


def _dynamic_moduli(
    files: list[UploadFile],
    ws: Path,
    settings: Settings,
    state: ThermoState,
    prony: PronyParams,
) -> tuple[list[Path], pd.DataFrame]:
    # Averaging happens on G(t), not on omega: two files' independently-Prony-fit frequency
    # grids aren't meaningfully combinable the way two files' G(t) lag-time series are, so the
    # Prony fit runs once, on the already-averaged G(t).
    paths, g_t = _stress_relaxation(files, ws, settings, state)
    df = rheology_service.dynamic_moduli_from_gt(
        g_t,
        method=prony.method,
        t_min=prony.t_min,
        t_cutoff=prony.t_cutoff,
        n_modes=prony.n_modes,
        n_omega=prony.n_omega,
        omega_min=prony.omega_min,
        omega_max=prony.omega_max,
    )
    return paths, df


@router.post("/stress-relaxation", response_model=TableResult)
def stress_relaxation(
    request: Request,
    files: FilesDep,
    state: ThermoStateDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
):
    paths, df = _stress_relaxation(files, ws, settings, state)
    result = table_result_from_df(
        df,
        title="Stress relaxation modulus",
        download_stem=f"{_stem(paths)}_G_t",
        max_rows=settings.max_table_rows,
        plot=G_T_PLOT,
        meta=_meta(paths),
    )
    return table_response(request, result, download_url="/api/rheology/stress-relaxation/download")


@router.post("/stress-relaxation/download", response_class=FileResponse, responses=FILE_RESPONSES)
def stress_relaxation_download(
    files: FilesDep, state: ThermoStateDep, settings: SettingsDep, fmt: FormatQuery = "csv"
):
    with download_workspace(settings) as ws:
        paths, df = _stress_relaxation(files, ws, settings, state)
        out = write_table(df, ws, f"{_stem(paths)}_G_t", fmt)
        return file_download(out, ws, media_type(fmt))


@router.post("/dynamic-moduli", response_model=TableResult)
def dynamic_moduli(
    request: Request,
    files: FilesDep,
    state: ThermoStateDep,
    prony: PronyParamsDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
):
    paths, df = _dynamic_moduli(files, ws, settings, state, prony)
    result = table_result_from_df(
        df,
        title="Dynamic moduli",
        download_stem=f"{_stem(paths)}_moduli",
        max_rows=settings.max_table_rows,
        plot=_moduli_plot(df),
        meta=_meta(paths),
    )
    return table_response(request, result, download_url="/api/rheology/dynamic-moduli/download")


@router.post("/dynamic-moduli/download", response_class=FileResponse, responses=FILE_RESPONSES)
def dynamic_moduli_download(
    files: FilesDep,
    state: ThermoStateDep,
    prony: PronyParamsDep,
    settings: SettingsDep,
    fmt: FormatQuery = "csv",
):
    with download_workspace(settings) as ws:
        paths, df = _dynamic_moduli(files, ws, settings, state, prony)
        out = write_table(df, ws, f"{_stem(paths)}_moduli", fmt)
        return file_download(out, ws, media_type(fmt))


@router.post("/stress-relaxation/start", response_model=StartResponse, status_code=202)
def stress_relaxation_start(
    files: FilesDep,
    state: ThermoStateDep,
    settings: SettingsDep,
    store: JobStoreDep,
    semaphore: JobSemaphoreDep,
) -> StartResponse:
    # calc_stress_relaxation has no loop worth reporting progress on (a single vectorized
    # pandas expression) — this still goes through the job/poll mechanism for the same
    # start-then-poll contract every calculation page uses, even though it finishes almost
    # instantly.
    ws, paths = start_workspace(
        settings, files, settings.max_upload_bytes, settings.max_files_per_request
    )

    def target(_progress_callback):
        frames = [
            rheology_service.stress_relaxation(p, state.volume, state.temperature) for p in paths
        ]
        df = merge.average_rheology(frames, ("G_GK", "G_FSR"))
        return table_result_from_df(
            df,
            title="Stress relaxation modulus",
            download_stem=f"{_stem(paths)}_G_t",
            max_rows=settings.max_table_rows,
            plot=G_T_PLOT,
            meta=_meta(paths),
        )

    job = run_job(store, ws, target, semaphore, settings.max_concurrent_jobs)
    return StartResponse(job_id=job.id)


@router.post("/dynamic-moduli/start", response_model=StartResponse, status_code=202)
def dynamic_moduli_start(
    files: FilesDep,
    state: ThermoStateDep,
    prony: PronyParamsDep,
    settings: SettingsDep,
    store: JobStoreDep,
    semaphore: JobSemaphoreDep,
) -> StartResponse:
    ws, paths = start_workspace(
        settings, files, settings.max_upload_bytes, settings.max_files_per_request
    )

    def target(_progress_callback):
        frames = [
            rheology_service.stress_relaxation(p, state.volume, state.temperature) for p in paths
        ]
        g_t = merge.average_rheology(frames, ("G_GK", "G_FSR"))
        df = rheology_service.dynamic_moduli_from_gt(
            g_t,
            method=prony.method,
            t_min=prony.t_min,
            t_cutoff=prony.t_cutoff,
            n_modes=prony.n_modes,
            n_omega=prony.n_omega,
            omega_min=prony.omega_min,
            omega_max=prony.omega_max,
        )
        return table_result_from_df(
            df,
            title="Dynamic moduli",
            download_stem=f"{_stem(paths)}_moduli",
            max_rows=settings.max_table_rows,
            plot=_moduli_plot(df),
            meta=_meta(paths),
        )

    job = run_job(store, ws, target, semaphore, settings.max_concurrent_jobs)
    return StartResponse(job_id=job.id)


def _merged_regimes_gt(regimes: list[RheologyRegime]) -> pd.DataFrame:
    x_arrays = [np.array(r.lag_time, dtype=float) for r in regimes]
    series = [
        {"G_GK": np.array(r.G_GK, dtype=float), "G_FSR": np.array(r.G_FSR, dtype=float)}
        for r in regimes
    ]
    x, cols = merge.merge_regimes(x_arrays, series)
    return pd.DataFrame({"lag_time": x, "G_GK": cols["G_GK"], "G_FSR": cols["G_FSR"]})


@router.post("/merge-regimes", response_model=TableResult)
def merge_regimes(request: Request, body: RheologyMergeRequest, settings: SettingsDep):
    """Combine several regimes' already-computed G(t) series into one curve.

    Takes no file: each regime's G(t) was already produced by stress-relaxation
    (client-orchestrated, one request per regime), so this step is pure numeric merging.
    """
    df = _merged_regimes_gt(body.regimes)
    result = table_result_from_df(
        df,
        title="Stress relaxation modulus (merged regimes)",
        download_stem=body.download_stem,
        max_rows=settings.max_table_rows,
        plot=G_T_PLOT,
        meta={"regimes": len(body.regimes)},
    )
    return table_response(request, result, download_url="/api/rheology/merge-regimes/download")


@router.post("/merge-regimes/download", response_class=FileResponse, responses=FILE_RESPONSES)
def merge_regimes_download(
    body: RheologyMergeRequest, settings: SettingsDep, fmt: FormatQuery = "csv"
):
    with download_workspace(settings) as ws:
        df = _merged_regimes_gt(body.regimes)
        out = write_table(df, ws, body.download_stem, fmt)
        return file_download(out, ws, media_type(fmt))


@router.post("/dynamic-moduli/from-regimes", response_model=TableResult)
def dynamic_moduli_from_regimes(
    request: Request, body: DynamicModuliFromRegimesRequest, settings: SettingsDep
):
    """Merge regimes' G(t), then fit the Prony series once on the combined curve.

    Not the frequency-domain equivalent of `merge_regimes`: two regimes' independently-fit
    omega grids aren't meaningfully concatenable, so the merge happens on G(t), one layer
    earlier, before the single Prony fit runs.
    """
    g_t = _merged_regimes_gt(body.regimes)
    df = rheology_service.dynamic_moduli_from_gt(
        g_t,
        method=body.method,
        t_min=body.t_min,
        t_cutoff=body.t_cutoff,
        n_modes=body.n_modes,
        n_omega=body.n_omega,
        omega_min=body.omega_min,
        omega_max=body.omega_max,
    )
    result = table_result_from_df(
        df,
        title="Dynamic moduli (merged regimes)",
        download_stem=body.download_stem,
        max_rows=settings.max_table_rows,
        plot=_moduli_plot(df),
        meta={"regimes": len(body.regimes)},
    )
    return table_response(
        request, result, download_url="/api/rheology/dynamic-moduli/from-regimes/download"
    )


@router.post(
    "/dynamic-moduli/from-regimes/download",
    response_class=FileResponse,
    responses=FILE_RESPONSES,
)
def dynamic_moduli_from_regimes_download(
    body: DynamicModuliFromRegimesRequest, settings: SettingsDep, fmt: FormatQuery = "csv"
):
    with download_workspace(settings) as ws:
        g_t = _merged_regimes_gt(body.regimes)
        df = rheology_service.dynamic_moduli_from_gt(
            g_t,
            method=body.method,
            t_min=body.t_min,
            t_cutoff=body.t_cutoff,
            n_modes=body.n_modes,
            n_omega=body.n_omega,
            omega_min=body.omega_min,
            omega_max=body.omega_max,
        )
        out = write_table(df, ws, body.download_stem, fmt)
        return file_download(out, ws, media_type(fmt))
