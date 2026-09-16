"""One generic job-status endpoint, shared by every calculation domain.

A job's `result` is always a `TableResult` regardless of which calculation produced it, so a
single endpoint has no information loss versus one per domain and avoids 8+ near-identical
route functions.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..jobs import JobStatus, JobStore, get_job_store
from ..schemas.common import TableResult

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

JobStoreDep = Annotated[JobStore, Depends(get_job_store)]


class JobStatusResponse(BaseModel):
    status: JobStatus
    progress: float
    message: str
    result: TableResult | None = None
    error: str | None = None


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str, store: JobStoreDep) -> JobStatusResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found or expired.")
    return JobStatusResponse(
        status=job.status,
        progress=job.progress,
        message=job.message,
        result=job.result,
        error=job.error,
    )
