"""Read a LAMMPS data file's chain topology and describe it, without calculating anything.

Mounted at /api/topology rather than under /api/dynamics on purpose: `main.py`'s body-size
middleware gives everything below /api/dynamics/ the trajectory ceiling, which is several
gigabytes, and this route accepts one small text file.
"""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from ..config import Settings, get_settings
from ..schemas.topology import TopologyPreview
from ..services import dynamics_service
from ..uploads import get_workspace, save_upload

router = APIRouter(prefix="/api/topology", tags=["topology"])

SettingsDep = Annotated[Settings, Depends(get_settings)]
WorkspaceDep = Annotated[Path, Depends(get_workspace)]
FileDep = Annotated[UploadFile, File()]


@router.post("/preview", response_model=TopologyPreview)
def preview(file: FileDep, settings: SettingsDep, ws: WorkspaceDep) -> TopologyPreview:
    """What chains this data file describes, so a wrong file is caught before a long run."""
    path = save_upload(file, ws, settings.max_topology_bytes)
    topology = dynamics_service.read_topology(path)
    return TopologyPreview(**dynamics_service.topology_stats(topology))
