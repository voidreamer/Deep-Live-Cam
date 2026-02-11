"""SQLAlchemy ORM models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), default="")
    google_id: Mapped[str | None] = mapped_column(String(255), default=None)
    tier: Mapped[str] = mapped_column(String(20), default="free")  # free | premium
    stripe_customer_id: Mapped[str | None] = mapped_column(String(255), default=None)
    stripe_subscription_id: Mapped[str | None] = mapped_column(String(255), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    jobs: Mapped[list[Job]] = relationship(back_populates="user")
    usage_records: Mapped[list[UsageRecord]] = relationship(back_populates="user")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    user_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("users.id"), nullable=True, default=None
    )
    job_type: Mapped[str] = mapped_column(String(10))  # image | video
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued | processing | done | failed
    priority: Mapped[int] = mapped_column(Integer, default=1)  # 0=premium, 1=free
    total_frames: Mapped[int] = mapped_column(Integer, default=0)
    processed_frames: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    result_path: Mapped[str | None] = mapped_column(String(500), default=None)
    options: Mapped[dict | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    user: Mapped[User | None] = relationship(back_populates="jobs")


class UsageRecord(Base):
    __tablename__ = "usage_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("users.id"), nullable=True, default=None
    )
    session_id: Mapped[str | None] = mapped_column(String(64), default=None)
    job_type: Mapped[str] = mapped_column(String(10))  # image | video
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    user: Mapped[User | None] = relationship(back_populates="usage_records")
