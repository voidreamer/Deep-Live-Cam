"""User profile and history endpoints."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user
from api.config import settings
from api.database import get_db
from api.models import Job, UsageRecord, User

router = APIRouter(prefix="/me", tags=["user"])


@router.get("")
async def me(user: User | None = Depends(get_current_user)):
    if user is None:
        raise HTTPException(status_code=401, detail="Not logged in")
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "tier": user.tier,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


@router.get("/usage")
async def usage(
    user: User | None = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if user is None:
        raise HTTPException(status_code=401, detail="Not logged in")

    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    image_count = (
        await db.execute(
            select(func.count(UsageRecord.id)).where(
                UsageRecord.user_id == user.id,
                UsageRecord.job_type == "image",
                UsageRecord.created_at >= today_start,
            )
        )
    ).scalar_one()

    video_count = (
        await db.execute(
            select(func.count(UsageRecord.id)).where(
                UsageRecord.user_id == user.id,
                UsageRecord.job_type == "video",
                UsageRecord.created_at >= today_start,
            )
        )
    ).scalar_one()

    if user.tier == "premium":
        img_limit, vid_limit = -1, -1
    else:
        img_limit = settings.free_image_swaps_per_day
        vid_limit = settings.free_video_swaps_per_day

    return {
        "tier": user.tier,
        "today": {
            "image_swaps": image_count,
            "video_swaps": video_count,
            "image_limit": img_limit,
            "video_limit": vid_limit,
        },
    }


@router.get("/history")
async def history(
    user: User | None = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if user is None:
        raise HTTPException(status_code=401, detail="Not logged in")

    # History retention: premium=30d, free=7d
    days = 30 if user.tier == "premium" else 7
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    result = await db.execute(
        select(Job)
        .where(Job.user_id == user.id, Job.created_at >= cutoff)
        .order_by(Job.created_at.desc())
        .limit(100)
    )
    jobs = result.scalars().all()

    return [
        {
            "id": j.id,
            "job_type": j.job_type,
            "status": j.status,
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "options": j.options,
        }
        for j in jobs
    ]
