"""Google OAuth authentication endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import auth_configured, create_jwt
from api.config import settings
from api.database import get_db
from api.models import User

router = APIRouter(prefix="/auth", tags=["auth"])


def _require_auth():
    if not auth_configured():
        raise HTTPException(status_code=404, detail="Authentication not configured")


@router.get("/google")
async def google_login(request: Request):
    _require_auth()
    from authlib.integrations.starlette_client import OAuth

    oauth = OAuth()
    oauth.register(
        name="google",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
    redirect_uri = str(request.url_for("google_callback"))
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/callback")
async def google_callback(request: Request, db: AsyncSession = Depends(get_db)):
    _require_auth()
    from authlib.integrations.starlette_client import OAuth

    oauth = OAuth()
    oauth.register(
        name="google",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

    token = await oauth.google.authorize_access_token(request)
    userinfo = token.get("userinfo", {})

    email = userinfo.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Could not get email from Google")

    # Upsert user
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if user is None:
        user = User(
            email=email,
            name=userinfo.get("name", ""),
            google_id=userinfo.get("sub"),
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
    elif not user.google_id:
        user.google_id = userinfo.get("sub")
        await db.commit()

    jwt_token = create_jwt(user.id, user.email)
    response = RedirectResponse(url="/")
    response.set_cookie(
        "dlc_token",
        jwt_token,
        httponly=True,
        samesite="lax",
        max_age=settings.jwt_expire_hours * 3600,
    )
    return response


@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/")
    response.delete_cookie("dlc_token")
    return response
