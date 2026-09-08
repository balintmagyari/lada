"""Routes for time correlation functions computed from an .npz trajectory."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/dynamics", tags=["dynamics"])
