"""Routes for time correlation functions computed from one or more .npz trajectories."""

import threading
from pathlib import Path
from typing import Annotated

import numpy as np
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
from ..schemas.dynamics import FrameTime, IsfParams, RouseMode, TopologyForm
from ..schemas.regimes import DynamicsMergeRequest
from ..services import dynamics_service, merge
from ..services.export_service import Format, media_type, table_result_from_df, write_table
from ..uploads import (
    cleanup_workspace,
    download_workspace,
    file_download,
    get_workspace,
    save_uploads,
)

router = APIRouter(prefix="/api/dynamics", tags=["dynamics"])

SettingsDep = Annotated[Settings, Depends(get_settings)]
WorkspaceDep = Annotated[Path, Depends(get_workspace)]
FilesDep = Annotated[list[UploadFile], File()]
FormatQuery = Annotated[Format, Query()]
TopologyDep = Annotated[TopologyForm, Depends()]
FrameTimeDep = Annotated[FrameTime, Depends()]
RouseModeDep = Annotated[RouseMode, Depends()]
IsfParamsDep = Annotated[IsfParams, Depends()]
JobStoreDep = Annotated[JobStore, Depends(get_job_store)]
JobSemaphoreDep = Annotated[threading.Semaphore, Depends(get_job_semaphore)]

Meta = dict[str, str | int | float]

LAG_LABEL = "Lag time"


def _stem(paths: list[Path]) -> str:
    """The download-stem prefix: a single file's own stem, or an "avgN" label for a repeat set."""
    return paths[0].stem if len(paths) == 1 else f"{paths[0].stem}_avg{len(paths)}"


def _inspect_and_validate(paths: list[Path], settings: Settings) -> tuple[list[int], int]:
    """Check that already-saved trajectories all describe the same system (atom count).

    Frame counts may differ between repeat runs (`merge.average_dynamics` truncates to the
    shortest), but the atom count has to match — it fixes what a given topology/index even means.
    """
    shapes = [
        dynamics_service.inspect_trajectory(
            p, settings.max_trajectory_frames, settings.max_trajectory_memory_bytes
        )
        for p in paths
    ]
    n_atoms = shapes[0][1]
    mismatched = [p.name for p, (_, a) in zip(paths, shapes, strict=True) if a != n_atoms]
    if mismatched:
        raise ValueError(
            f"All files must have the same number of atoms ({n_atoms}, from "
            f"'{paths[0].name}'); {', '.join(mismatched)} do not match."
        )
    return [f for f, _ in shapes], n_atoms


def _load_many(
    files: list[UploadFile], ws: Path, settings: Settings
) -> tuple[list[Path], list[int], int]:
    """Save every upload, then check they all describe the same system (atom count)."""
    paths = save_uploads(files, ws, settings.max_trajectory_bytes, settings.max_files_per_request)
    n_frames, n_atoms = _inspect_and_validate(paths, settings)
    return paths, n_frames, n_atoms


def _segment_acf(
    files: list[UploadFile], ws: Path, settings: Settings, topology: TopologyForm, timing: FrameTime
) -> tuple[list[Path], pd.DataFrame, Meta]:
    paths, n_frames, n_atoms = _load_many(files, ws, settings)
    pairs = topology.segment_pairs(n_atoms)
    dynamics_service.check_indices(pairs, n_atoms)
    frames = [dynamics_service.segment_acf(p, pairs, timing.time_per_frame) for p in paths]
    df = merge.average_dynamics(frames, "acf")
    meta: Meta = {
        "files": ", ".join(p.name for p in paths),
        "n_files": len(paths),
        "frames": len(df),
        "atoms": n_atoms,
        "chains": len(pairs),
    }
    return paths, df, meta


def _rouse_mode_acf(
    files: list[UploadFile],
    ws: Path,
    settings: Settings,
    topology: TopologyForm,
    timing: FrameTime,
    mode: RouseMode,
) -> tuple[list[Path], pd.DataFrame, Meta]:
    paths, n_frames, n_atoms = _load_many(files, ws, settings)
    chains = topology.chain_indices(n_atoms)
    dynamics_service.check_indices(chains, n_atoms)
    frames = [
        dynamics_service.rouse_mode_acf(p, chains, mode.p, timing.time_per_frame) for p in paths
    ]
    df = merge.average_dynamics(frames, "acf")
    meta: Meta = {
        "files": ", ".join(p.name for p in paths),
        "n_files": len(paths),
        "frames": len(df),
        "atoms": n_atoms,
        "chains": len(chains),
    }
    return paths, df, meta


def _isf(
    files: list[UploadFile], ws: Path, settings: Settings, timing: FrameTime, params: IsfParams
) -> tuple[list[Path], pd.DataFrame, Meta]:
    paths, n_frames, n_atoms = _load_many(files, ws, settings)
    frames = [
        dynamics_service.isf(p, timing.time_per_frame, params.q_magnitude, params.n_vectors)
        for p in paths
    ]
    df = merge.average_dynamics(frames, "isf")
    meta: Meta = {
        "files": ", ".join(p.name for p in paths),
        "n_files": len(paths),
        "frames": len(df),
        "atoms": n_atoms,
    }
    return paths, df, meta


@router.post("/segment-acf", response_model=TableResult)
def segment_acf(
    request: Request,
    files: FilesDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
    topology: TopologyDep,
    timing: FrameTimeDep,
):
    paths, df, meta = _segment_acf(files, ws, settings, topology, timing)
    result = table_result_from_df(
        df,
        title="End-to-end vector autocorrelation",
        download_stem=f"{_stem(paths)}_segment_acf",
        max_rows=settings.max_table_rows,
        plot=PlotSpec(x="lag_time", y=["acf"], x_label=LAG_LABEL, y_label="C(t)"),
        meta=meta,
    )
    return table_response(request, result, download_url="/api/dynamics/segment-acf/download")


@router.post("/segment-acf/download", response_class=FileResponse, responses=FILE_RESPONSES)
def segment_acf_download(
    files: FilesDep,
    settings: SettingsDep,
    topology: TopologyDep,
    timing: FrameTimeDep,
    fmt: FormatQuery = "csv",
):
    with download_workspace(settings) as ws:
        paths, df, _ = _segment_acf(files, ws, settings, topology, timing)
        out = write_table(df, ws, f"{_stem(paths)}_segment_acf", fmt)
        return file_download(out, ws, media_type(fmt))


@router.post("/rouse-mode", response_model=TableResult)
def rouse_mode(
    request: Request,
    files: FilesDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
    topology: TopologyDep,
    timing: FrameTimeDep,
    mode: RouseModeDep,
):
    paths, df, meta = _rouse_mode_acf(files, ws, settings, topology, timing, mode)
    result = table_result_from_df(
        df,
        title=f"Rouse mode {mode.p} autocorrelation",
        download_stem=f"{_stem(paths)}_rouse_p{mode.p}",
        max_rows=settings.max_table_rows,
        plot=PlotSpec(x="lag_time", y=["acf"], x_label=LAG_LABEL, y_label=f"C_{mode.p}(t)"),
        meta=meta,
    )
    return table_response(request, result, download_url="/api/dynamics/rouse-mode/download")


@router.post("/rouse-mode/download", response_class=FileResponse, responses=FILE_RESPONSES)
def rouse_mode_download(
    files: FilesDep,
    settings: SettingsDep,
    topology: TopologyDep,
    timing: FrameTimeDep,
    mode: RouseModeDep,
    fmt: FormatQuery = "csv",
):
    with download_workspace(settings) as ws:
        paths, df, _ = _rouse_mode_acf(files, ws, settings, topology, timing, mode)
        out = write_table(df, ws, f"{_stem(paths)}_rouse_p{mode.p}", fmt)
        return file_download(out, ws, media_type(fmt))


@router.post("/isf", response_model=TableResult)
def isf(
    request: Request,
    files: FilesDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
    timing: FrameTimeDep,
    params: IsfParamsDep,
):
    paths, df, meta = _isf(files, ws, settings, timing, params)
    result = table_result_from_df(
        df,
        title="Intermediate scattering function",
        download_stem=f"{_stem(paths)}_isf",
        max_rows=settings.max_table_rows,
        plot=PlotSpec(x="lag_time", y=["isf"], x_label=LAG_LABEL, y_label="F(q,t) / F(q,0)"),
        meta=meta,
    )
    return table_response(request, result, download_url="/api/dynamics/isf/download")


@router.post("/isf/download", response_class=FileResponse, responses=FILE_RESPONSES)
def isf_download(
    files: FilesDep,
    settings: SettingsDep,
    timing: FrameTimeDep,
    params: IsfParamsDep,
    fmt: FormatQuery = "csv",
):
    with download_workspace(settings) as ws:
        paths, df, _ = _isf(files, ws, settings, timing, params)
        out = write_table(df, ws, f"{_stem(paths)}_isf", fmt)
        return file_download(out, ws, media_type(fmt))


@router.post("/segment-acf/start", response_model=StartResponse, status_code=202)
def segment_acf_start(
    files: FilesDep,
    settings: SettingsDep,
    topology: TopologyDep,
    timing: FrameTimeDep,
    store: JobStoreDep,
    semaphore: JobSemaphoreDep,
) -> StartResponse:
    ws, paths = start_workspace(
        settings, files, settings.max_trajectory_bytes, settings.max_files_per_request
    )
    try:
        _, n_atoms = _inspect_and_validate(paths, settings)
        pairs = topology.segment_pairs(n_atoms)
        dynamics_service.check_indices(pairs, n_atoms)
    except Exception:
        cleanup_workspace(ws)
        raise

    def target(progress_callback):
        frames = [
            dynamics_service.segment_acf(
                p,
                pairs,
                timing.time_per_frame,
                progress_callback=per_file_progress(progress_callback, i, len(paths)),
            )
            for i, p in enumerate(paths)
        ]
        df = merge.average_dynamics(frames, "acf")
        meta: Meta = {
            "files": ", ".join(p.name for p in paths),
            "n_files": len(paths),
            "frames": len(df),
            "atoms": n_atoms,
            "chains": len(pairs),
        }
        return table_result_from_df(
            df,
            title="End-to-end vector autocorrelation",
            download_stem=f"{_stem(paths)}_segment_acf",
            max_rows=settings.max_table_rows,
            plot=PlotSpec(x="lag_time", y=["acf"], x_label=LAG_LABEL, y_label="C(t)"),
            meta=meta,
        )

    job = run_job(store, ws, target, semaphore, settings.max_concurrent_jobs)
    return StartResponse(job_id=job.id)


@router.post("/rouse-mode/start", response_model=StartResponse, status_code=202)
def rouse_mode_start(
    files: FilesDep,
    settings: SettingsDep,
    topology: TopologyDep,
    timing: FrameTimeDep,
    mode: RouseModeDep,
    store: JobStoreDep,
    semaphore: JobSemaphoreDep,
) -> StartResponse:
    ws, paths = start_workspace(
        settings, files, settings.max_trajectory_bytes, settings.max_files_per_request
    )
    try:
        _, n_atoms = _inspect_and_validate(paths, settings)
        chains = topology.chain_indices(n_atoms)
        dynamics_service.check_indices(chains, n_atoms)
    except Exception:
        cleanup_workspace(ws)
        raise

    def target(progress_callback):
        frames = [
            dynamics_service.rouse_mode_acf(
                p,
                chains,
                mode.p,
                timing.time_per_frame,
                progress_callback=per_file_progress(progress_callback, i, len(paths)),
            )
            for i, p in enumerate(paths)
        ]
        df = merge.average_dynamics(frames, "acf")
        meta: Meta = {
            "files": ", ".join(p.name for p in paths),
            "n_files": len(paths),
            "frames": len(df),
            "atoms": n_atoms,
            "chains": len(chains),
        }
        return table_result_from_df(
            df,
            title=f"Rouse mode {mode.p} autocorrelation",
            download_stem=f"{_stem(paths)}_rouse_p{mode.p}",
            max_rows=settings.max_table_rows,
            plot=PlotSpec(x="lag_time", y=["acf"], x_label=LAG_LABEL, y_label=f"C_{mode.p}(t)"),
            meta=meta,
        )

    job = run_job(store, ws, target, semaphore, settings.max_concurrent_jobs)
    return StartResponse(job_id=job.id)


@router.post("/isf/start", response_model=StartResponse, status_code=202)
def isf_start(
    files: FilesDep,
    settings: SettingsDep,
    timing: FrameTimeDep,
    params: IsfParamsDep,
    store: JobStoreDep,
    semaphore: JobSemaphoreDep,
) -> StartResponse:
    ws, paths = start_workspace(
        settings, files, settings.max_trajectory_bytes, settings.max_files_per_request
    )
    try:
        _, n_atoms = _inspect_and_validate(paths, settings)
    except Exception:
        cleanup_workspace(ws)
        raise

    def target(progress_callback):
        frames = [
            dynamics_service.isf(
                p,
                timing.time_per_frame,
                params.q_magnitude,
                params.n_vectors,
                progress_callback=per_file_progress(progress_callback, i, len(paths)),
            )
            for i, p in enumerate(paths)
        ]
        df = merge.average_dynamics(frames, "isf")
        meta: Meta = {
            "files": ", ".join(p.name for p in paths),
            "n_files": len(paths),
            "frames": len(df),
            "atoms": n_atoms,
        }
        return table_result_from_df(
            df,
            title="Intermediate scattering function",
            download_stem=f"{_stem(paths)}_isf",
            max_rows=settings.max_table_rows,
            plot=PlotSpec(x="lag_time", y=["isf"], x_label=LAG_LABEL, y_label="F(q,t) / F(q,0)"),
            meta=meta,
        )

    job = run_job(store, ws, target, semaphore, settings.max_concurrent_jobs)
    return StartResponse(job_id=job.id)


def _merged_regimes_df(body: DynamicsMergeRequest) -> pd.DataFrame:
    x_arrays = [np.array(r.lag_time, dtype=float) for r in body.regimes]
    series = [{body.value_column: np.array(r.value, dtype=float)} for r in body.regimes]
    x, cols = merge.merge_regimes(x_arrays, series)
    return pd.DataFrame({"lag_time": x, body.value_column: cols[body.value_column]})


@router.post("/merge-regimes", response_model=TableResult)
def merge_regimes(request: Request, body: DynamicsMergeRequest, settings: SettingsDep):
    """Combine several regimes' already-computed (lag_time, value) series into one curve.

    Takes no file: each regime's series was already produced by segment-acf/rouse-mode/isf
    (client-orchestrated, one request per regime), so this step is pure numeric merging.
    """
    df = _merged_regimes_df(body)
    result = table_result_from_df(
        df,
        title=body.title,
        download_stem=body.download_stem,
        max_rows=settings.max_table_rows,
        plot=PlotSpec(x="lag_time", y=[body.value_column], x_label=LAG_LABEL, y_label=body.y_label),
        meta={"regimes": len(body.regimes)},
    )
    return table_response(request, result, download_url="/api/dynamics/merge-regimes/download")


@router.post("/merge-regimes/download", response_class=FileResponse, responses=FILE_RESPONSES)
def merge_regimes_download(
    body: DynamicsMergeRequest, settings: SettingsDep, fmt: FormatQuery = "csv"
):
    with download_workspace(settings) as ws:
        df = _merged_regimes_df(body)
        out = write_table(df, ws, body.download_stem, fmt)
        return file_download(out, ws, media_type(fmt))
