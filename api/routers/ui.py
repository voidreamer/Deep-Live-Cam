"""Serve the web UI."""

from __future__ import annotations

import os

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(include_in_schema=False)

_template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api", "templates")

# Resolve relative to this file
_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "..", "templates", "index.html")


@router.get("/", response_class=HTMLResponse)
async def root():
    path = os.path.normpath(_TEMPLATE_PATH)
    with open(path) as f:
        return HTMLResponse(f.read())
