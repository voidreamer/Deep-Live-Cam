"""Job status and download endpoints."""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.models import Job
from api.queue import job_queue
from api.storage import result_path

router = APIRouter(tags=["jobs"])


@router.get("/job/{job_id}")
async def job_status(job_id: str, db: AsyncSession = Depends(get_db)):
    # Check in-memory queue state first (most up-to-date for active jobs)
    state = job_queue.get_state(job_id)
    if state:
        return {
            "status": state["status"],
            "total_frames": state.get("total_frames", 0),
            "processed_frames": state.get("processed_frames", 0),
            "error": state.get("error"),
        }

    # Fall back to DB
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "status": job.status,
        "total_frames": job.total_frames,
        "processed_frames": job.processed_frames,
        "error": job.error,
    }


@router.get("/job/{job_id}/download")
async def job_download(job_id: str, db: AsyncSession = Depends(get_db)):
    # Check in-memory state for result path
    state = job_queue.get_state(job_id)

    if state:
        if state["status"] != "done":
            raise HTTPException(
                status_code=400, detail=f"Job is not done (status: {state['status']})"
            )
        path = state.get("result_path") or result_path(job_id, ".mp4")
    else:
        result = await db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != "done":
            raise HTTPException(
                status_code=400, detail=f"Job is not done (status: {job.status})"
            )
        path = job.result_path or result_path(job_id, ".mp4")

    if not os.path.exists(path):
        raise HTTPException(status_code=500, detail="Result file missing")

    # Determine media type from extension
    ext = os.path.splitext(path)[1].lower()
    media_type = "video/mp4" if ext == ".mp4" else "image/jpeg"
    filename = f"result{ext}"

    return FileResponse(
        path,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
