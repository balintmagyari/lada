"""Routes for relaxation modulus and dynamic moduli calculations."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/rheology", tags=["rheology"])
