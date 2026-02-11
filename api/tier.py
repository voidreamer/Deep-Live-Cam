"""Usage tier enforcement."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings
from api.models import UsageRecord, User


def get_max_video_bytes(user: User | None) -> int:
    """Return the max video upload size for the user's tier."""
    if user is None:
        return settings.max_video_bytes_free
    if user.tier == "premium":
        return settings.max_video_bytes_premium
    return settings.max_video_bytes_logged_in


async def check_usage_limit(
    request: Request,
    user: User | None,
    job_type: str,
    db: AsyncSession,
) -> None:
    """Raise 429 if the user/session has exceeded their daily limit."""
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    if user and user.tier == "premium":
        return  # unlimited

    # Determine limit
    if user:
        limit = (
            settings.free_image_swaps_per_day
            if job_type == "image"
            else settings.free_video_swaps_per_day
        )
        count_query = select(func.count(UsageRecord.id)).where(
            UsageRecord.user_id == user.id,
            UsageRecord.job_type == job_type,
            UsageRecord.created_at >= today_start,
        )
    else:
        limit = (
            settings.anon_image_swaps_per_day
            if job_type == "image"
            else settings.anon_video_swaps_per_day
        )
        # Track anonymous users by session cookie
        session_id = request.cookies.get("dlc_session")
        if not session_id:
            return  # no session → can't track, allow (session set on response)
        count_query = select(func.count(UsageRecord.id)).where(
            UsageRecord.session_id == session_id,
            UsageRecord.user_id.is_(None),
            UsageRecord.job_type == job_type,
            UsageRecord.created_at >= today_start,
        )

    count = (await db.execute(count_query)).scalar_one()

    if limit != -1 and count >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Daily {job_type} swap limit reached ({limit}). Upgrade for unlimited access.",
            headers={"Retry-After": "86400"},
        )
