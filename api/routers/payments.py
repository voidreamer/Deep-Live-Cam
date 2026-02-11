"""Stripe payment endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user
from api.config import settings
from api.database import get_db
from api.models import User

router = APIRouter(tags=["payments"])


def _stripe_configured() -> bool:
    return bool(settings.stripe_secret_key and settings.stripe_price_id)


def _get_stripe():
    if not _stripe_configured():
        raise HTTPException(status_code=404, detail="Payments not configured")
    import stripe
    stripe.api_key = settings.stripe_secret_key
    return stripe


@router.get("/checkout")
async def checkout(
    request: Request,
    user: User | None = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if user is None:
        raise HTTPException(status_code=401, detail="Login required to upgrade")
    stripe = _get_stripe()

    # Create or reuse Stripe customer
    if not user.stripe_customer_id:
        customer = stripe.Customer.create(email=user.email, metadata={"user_id": user.id})
        user.stripe_customer_id = customer.id
        await db.commit()

    base_url = str(request.base_url).rstrip("/")
    session = stripe.checkout.Session.create(
        customer=user.stripe_customer_id,
        mode="subscription",
        line_items=[{"price": settings.stripe_price_id, "quantity": 1}],
        success_url=f"{base_url}/?upgraded=1",
        cancel_url=f"{base_url}/",
    )
    return RedirectResponse(session.url, status_code=303)


@router.post("/stripe/webhook")
async def stripe_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    stripe = _get_stripe()
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig, settings.stripe_webhook_secret
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        customer_id = session.get("customer")
        subscription_id = session.get("subscription")
        if customer_id:
            result = await db.execute(
                select(User).where(User.stripe_customer_id == customer_id)
            )
            user = result.scalar_one_or_none()
            if user:
                user.tier = "premium"
                user.stripe_subscription_id = subscription_id
                await db.commit()

    elif event["type"] in ("customer.subscription.deleted", "invoice.payment_failed"):
        obj = event["data"]["object"]
        customer_id = obj.get("customer")
        if customer_id:
            result = await db.execute(
                select(User).where(User.stripe_customer_id == customer_id)
            )
            user = result.scalar_one_or_none()
            if user:
                user.tier = "free"
                user.stripe_subscription_id = None
                await db.commit()

    return {"status": "ok"}


@router.get("/billing")
async def billing(
    user: User | None = Depends(get_current_user),
):
    if user is None:
        raise HTTPException(status_code=401, detail="Login required")
    if not user.stripe_customer_id:
        raise HTTPException(status_code=400, detail="No billing account found")
    stripe = _get_stripe()
    session = stripe.billing_portal.Session.create(
        customer=user.stripe_customer_id,
        return_url="/",
    )
    return RedirectResponse(session.url, status_code=303)
