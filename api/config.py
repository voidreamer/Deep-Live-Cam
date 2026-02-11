"""Application configuration via environment variables."""

from __future__ import annotations

import secrets
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Database ---
    database_url: str = "sqlite+aiosqlite:///./data/dlc.db"

    # --- Storage ---
    storage_path: str = "./data/results"
    result_ttl_hours: int = 24

    # --- Auth ---
    jwt_secret: str = secrets.token_urlsafe(32)
    jwt_algorithm: str = "HS256"
    jwt_expire_hours: int = 72
    google_client_id: str = ""
    google_client_secret: str = ""

    # --- Stripe ---
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_price_id: str = ""

    # --- Limits ---
    max_image_bytes: int = 10 * 1024 * 1024  # 10 MB
    max_video_bytes_free: int = 25 * 1024 * 1024  # 25 MB
    max_video_bytes_logged_in: int = 50 * 1024 * 1024  # 50 MB
    max_video_bytes_premium: int = 100 * 1024 * 1024  # 100 MB

    # Tier: anonymous
    anon_image_swaps_per_day: int = 5
    anon_video_swaps_per_day: int = 1
    # Tier: free (logged in)
    free_image_swaps_per_day: int = 10
    free_video_swaps_per_day: int = 3
    # Tier: premium (unlimited, -1 sentinel)
    premium_image_swaps_per_day: int = -1
    premium_video_swaps_per_day: int = -1

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

# Ensure storage directory exists
Path(settings.storage_path).mkdir(parents=True, exist_ok=True)
