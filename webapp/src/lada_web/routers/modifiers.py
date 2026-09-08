"""Routes that rewrite a LAMMPS data file and hand back the modified file."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/modifiers", tags=["modifiers"])
