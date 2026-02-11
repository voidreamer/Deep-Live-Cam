"""Face swap endpoints: image and video."""

from __future__ import annotations

import io
import os
import tempfile
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user
from api.config import settings
from api.database import get_db
from api.models import Job, UsageRecord, User
from api.queue import job_queue
from api.storage import result_path, save_result
from api.tier import check_usage_limit, get_max_video_bytes

router = APIRouter(tags=["swap"])


def _decode_image(data: bytes, label: str) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail=f"Could not decode {label} image")
    return img


@router.post("/swap")
async def swap(
    request: Request,
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    many_faces: bool = Query(False),
    enhance: bool = Query(False),
    user: User | None = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_usage_limit(request, user, "image", db)

    source_bytes = await source.read()
    if len(source_bytes) > settings.max_image_bytes:
        raise HTTPException(status_code=400, detail="Source file exceeds size limit")

    target_bytes = await target.read()
    if len(target_bytes) > settings.max_image_bytes:
        raise HTTPException(status_code=400, detail="Target file exceeds size limit")

    # Premium-only enhancement
    if enhance and (user is None or user.tier != "premium"):
        raise HTTPException(status_code=403, detail="Face enhancement requires a premium subscription")

    source_img = _decode_image(source_bytes, "source")
    target_img = _decode_image(target_bytes, "target")

    from modules.face_analyser import get_many_faces, get_one_face
    from modules.processors.frame.face_swapper import swap_face

    source_face = get_one_face(source_img)
    if source_face is None:
        raise HTTPException(status_code=400, detail="No face detected in source image")

    if many_faces:
        target_faces = get_many_faces(target_img)
        if not target_faces:
            raise HTTPException(status_code=400, detail="No faces detected in target image")
    else:
        one = get_one_face(target_img)
        if one is None:
            raise HTTPException(status_code=400, detail="No face detected in target image")
        target_faces = [one]

    try:
        result = target_img
        for tf in target_faces:
            result = swap_face(source_face, tf, result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Face swap failed: {exc}")

    if enhance:
        try:
            from modules.processors.frame.face_enhancer import enhance_face
            result = enhance_face(result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Face enhancement failed: {exc}")

    ok, buf = cv2.imencode(".jpg", result)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode result image")

    # Record usage + persist job
    job_id = uuid.uuid4().hex
    result_file = save_result(job_id, buf.tobytes(), ".jpg")

    job = Job(
        id=job_id, user_id=user.id if user else None,
        job_type="image", status="done",
        priority=0 if user and user.tier == "premium" else 1,
        options={"many_faces": many_faces, "enhance": enhance},
        result_path=result_file,
    )
    usage = UsageRecord(
        user_id=user.id if user else None,
        session_id=request.cookies.get("dlc_session"),
        job_type="image",
    )
    db.add(job)
    db.add(usage)
    await db.commit()

    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")


@router.post("/swap/video")
async def swap_video(
    request: Request,
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    many_faces: bool = Query(False),
    enhance: bool = Query(False),
    user: User | None = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_usage_limit(request, user, "video", db)

    source_bytes = await source.read()
    if len(source_bytes) > settings.max_image_bytes:
        raise HTTPException(status_code=400, detail="Source file exceeds size limit")

    max_video = get_max_video_bytes(user)
    target_bytes = await target.read()
    if len(target_bytes) > max_video:
        mb = max_video // (1024 * 1024)
        raise HTTPException(status_code=400, detail=f"Target video exceeds {mb} MB limit")

    if enhance and (user is None or user.tier != "premium"):
        raise HTTPException(status_code=403, detail="Face enhancement requires a premium subscription")

    source_img = _decode_image(source_bytes, "source")

    from modules.face_analyser import get_one_face
    source_face = get_one_face(source_img)
    if source_face is None:
        raise HTTPException(status_code=400, detail="No face detected in source image")

    suffix = os.path.splitext(target.filename or "video.mp4")[1] or ".mp4"
    tmp_in = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_in.write(target_bytes)
    tmp_in.close()

    job_id = uuid.uuid4().hex
    priority = 0 if user and user.tier == "premium" else 1

    job = Job(
        id=job_id, user_id=user.id if user else None,
        job_type="video", status="queued", priority=priority,
        options={"many_faces": many_faces, "enhance": enhance},
    )
    usage = UsageRecord(
        user_id=user.id if user else None,
        session_id=request.cookies.get("dlc_session"),
        job_type="video",
    )
    db.add(job)
    db.add(usage)
    await db.commit()

    job_queue.enqueue(job_id, priority, {
        "source_face": source_face,
        "tmp_in": tmp_in.name,
        "many_faces": many_faces,
        "enhance": enhance,
    })

    return {"job_id": job_id}
