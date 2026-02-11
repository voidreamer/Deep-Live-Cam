"""Authentication: Google OAuth + JWT."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import Cookie, Depends, Request
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings
from api.database import get_db
from api.models import User


def create_jwt(user_id: str, email: str) -> str:
    """Create a signed JWT for the given user."""
    expire = datetime.now(timezone.utc) + timedelta(hours=settings.jwt_expire_hours)
    payload = {"sub": user_id, "email": email, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_jwt(token: str) -> dict | None:
    """Decode and verify a JWT. Returns payload or None."""
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError:
        return None


async def get_current_user(
    request: Request,
    dlc_token: str | None = Cookie(default=None),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """FastAPI dependency: returns the logged-in User or None (anonymous)."""
    token = dlc_token
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
    if not token:
        return None
    payload = decode_jwt(token)
    if not payload or "sub" not in payload:
        return None
    result = await db.execute(select(User).where(User.id == payload["sub"]))
    return result.scalar_one_or_none()


def auth_configured() -> bool:
    """Check if Google OAuth credentials are configured."""
    return bool(settings.google_client_id and settings.google_client_secret)
