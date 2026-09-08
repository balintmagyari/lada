"""Routes for time correlation functions computed from an .npz trajectory."""

from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import FileResponse

from ..config import Settings, get_settings
from ..rendering import table_response
from ..schemas.common import FILE_RESPONSES, PlotSpec, TableResult
from ..schemas.dynamics import FrameTime, IsfParams, RouseMode, TopologyForm
from ..services import dynamics_service
from ..services.export_service import Format, media_type, table_result_from_df, write_table
from ..uploads import download_workspace, file_download, get_workspace, save_upload

router = APIRouter(prefix="/api/dynamics", tags=["dynamics"])

SettingsDep = Annotated[Settings, Depends(get_settings)]
WorkspaceDep = Annotated[Path, Depends(get_workspace)]
FileDep = Annotated[UploadFile, File()]
FormatQuery = Annotated[Format, Query()]
TopologyDep = Annotated[TopologyForm, Depends()]
FrameTimeDep = Annotated[FrameTime, Depends()]
RouseModeDep = Annotated[RouseMode, Depends()]
IsfParamsDep = Annotated[IsfParams, Depends()]

Meta = dict[str, str | int | float]

LAG_LABEL = "Lag time"


def _load(file: UploadFile, ws: Path, settings: Settings) -> tuple[Path, int, int]:
    """Save the upload, then read the trajectory shape before any array is loaded."""
    path = save_upload(file, ws, settings.max_trajectory_bytes)
    n_frames, n_atoms = dynamics_service.inspect_trajectory(
        path, settings.max_trajectory_frames, settings.max_trajectory_memory_bytes
    )
    return path, n_frames, n_atoms


def _segment_acf(
    file: UploadFile, ws: Path, settings: Settings, topology: TopologyForm, timing: FrameTime
) -> tuple[Path, pd.DataFrame, Meta]:
    path, n_frames, n_atoms = _load(file, ws, settings)
    pairs = topology.segment_pairs(n_atoms)
    dynamics_service.check_indices(pairs, n_atoms)
    df = dynamics_service.segment_acf(path, pairs, timing.time_per_frame)
    meta: Meta = {
        "file": path.name,
        "frames": n_frames,
        "atoms": n_atoms,
        "chains": len(pairs),
    }
    return path, df, meta


def _rouse_mode_acf(
    file: UploadFile,
    ws: Path,
    settings: Settings,
    topology: TopologyForm,
    timing: FrameTime,
    mode: RouseMode,
) -> tuple[Path, pd.DataFrame, Meta]:
    path, n_frames, n_atoms = _load(file, ws, settings)
    chains = topology.chain_indices(n_atoms)
    dynamics_service.check_indices(chains, n_atoms)
    df = dynamics_service.rouse_mode_acf(path, chains, mode.p, timing.time_per_frame)
    meta: Meta = {
        "file": path.name,
        "frames": n_frames,
        "atoms": n_atoms,
        "chains": len(chains),
    }
    return path, df, meta


def _isf(
    file: UploadFile, ws: Path, settings: Settings, timing: FrameTime, params: IsfParams
) -> tuple[Path, pd.DataFrame, Meta]:
    path, n_frames, n_atoms = _load(file, ws, settings)
    df = dynamics_service.isf(path, timing.time_per_frame, params.q_magnitude, params.n_vectors)
    meta: Meta = {
        "file": path.name,
        "frames": n_frames,
        "atoms": n_atoms,
    }
    return path, df, meta


@router.post("/segment-acf", response_model=TableResult)
def segment_acf(
    request: Request,
    file: FileDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
    topology: TopologyDep,
    timing: FrameTimeDep,
):
    path, df, meta = _segment_acf(file, ws, settings, topology, timing)
    result = table_result_from_df(
        df,
        title="End-to-end vector autocorrelation",
        download_stem=f"{path.stem}_segment_acf",
        max_rows=settings.max_table_rows,
        plot=PlotSpec(x="lag_time", y=["acf"], x_label=LAG_LABEL, y_label="C(t)"),
        meta=meta,
    )
    return table_response(request, result, download_url="/api/dynamics/segment-acf/download")


@router.post("/segment-acf/download", response_class=FileResponse, responses=FILE_RESPONSES)
def segment_acf_download(
    file: FileDep,
    settings: SettingsDep,
    topology: TopologyDep,
    timing: FrameTimeDep,
    fmt: FormatQuery = "csv",
):
    with download_workspace(settings) as ws:
        path, df, _ = _segment_acf(file, ws, settings, topology, timing)
        out = write_table(df, ws, f"{path.stem}_segment_acf", fmt)
        return file_download(out, ws, media_type(fmt))


@router.post("/rouse-mode", response_model=TableResult)
def rouse_mode(
    request: Request,
    file: FileDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
    topology: TopologyDep,
    timing: FrameTimeDep,
    mode: RouseModeDep,
):
    path, df, meta = _rouse_mode_acf(file, ws, settings, topology, timing, mode)
    result = table_result_from_df(
        df,
        title=f"Rouse mode {mode.p} autocorrelation",
        download_stem=f"{path.stem}_rouse_p{mode.p}",
        max_rows=settings.max_table_rows,
        plot=PlotSpec(x="lag_time", y=["acf"], x_label=LAG_LABEL, y_label=f"C_{mode.p}(t)"),
        meta=meta,
    )
    return table_response(request, result, download_url="/api/dynamics/rouse-mode/download")


@router.post("/rouse-mode/download", response_class=FileResponse, responses=FILE_RESPONSES)
def rouse_mode_download(
    file: FileDep,
    settings: SettingsDep,
    topology: TopologyDep,
    timing: FrameTimeDep,
    mode: RouseModeDep,
    fmt: FormatQuery = "csv",
):
    with download_workspace(settings) as ws:
        path, df, _ = _rouse_mode_acf(file, ws, settings, topology, timing, mode)
        out = write_table(df, ws, f"{path.stem}_rouse_p{mode.p}", fmt)
        return file_download(out, ws, media_type(fmt))


@router.post("/isf", response_model=TableResult)
def isf(
    request: Request,
    file: FileDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
    timing: FrameTimeDep,
    params: IsfParamsDep,
):
    path, df, meta = _isf(file, ws, settings, timing, params)
    result = table_result_from_df(
        df,
        title="Intermediate scattering function",
        download_stem=f"{path.stem}_isf",
        max_rows=settings.max_table_rows,
        plot=PlotSpec(x="lag_time", y=["isf"], x_label=LAG_LABEL, y_label="F(q,t) / F(q,0)"),
        meta=meta,
    )
    return table_response(request, result, download_url="/api/dynamics/isf/download")


@router.post("/isf/download", response_class=FileResponse, responses=FILE_RESPONSES)
def isf_download(
    file: FileDep,
    settings: SettingsDep,
    timing: FrameTimeDep,
    params: IsfParamsDep,
    fmt: FormatQuery = "csv",
):
    with download_workspace(settings) as ws:
        path, df, _ = _isf(file, ws, settings, timing, params)
        out = write_table(df, ws, f"{path.stem}_isf", fmt)
        return file_download(out, ws, media_type(fmt))
