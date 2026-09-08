"""Routes that rewrite a LAMMPS data file and hand back the modified file."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from ..config import Settings, get_settings
from ..schemas.modifiers import EndBeadParams
from ..services.modifiers_service import rewrite_end_beads_file
from ..uploads import download_workspace, file_download, save_upload

router = APIRouter(prefix="/api/modifiers", tags=["modifiers"])

SettingsDep = Annotated[Settings, Depends(get_settings)]
FileDep = Annotated[UploadFile, File()]
ParamsDep = Annotated[EndBeadParams, Depends()]


@router.post("/rewrite-end-beads")
def rewrite_end_beads_route(file: FileDep, params: ParamsDep, settings: SettingsDep):
    with download_workspace(settings) as ws:
        path = save_upload(file, ws, settings.max_upload_bytes)
        out = rewrite_end_beads_file(
            path,
            ws / f"{path.stem}_endbeads.data",
            new_end_type=params.new_end_type,
            base_type=params.base_type,
        )
        return file_download(out, ws, "text/plain")
