"""Deep-Live-Cam SaaS API — FastAPI application factory."""

from __future__ import annotations

import asyncio
import platform
import sys
import types
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

# Stub modules.core so face_swapper/face_enhancer don't pull in tensorflow
if "modules.core" not in sys.modules:
    _core_stub = types.ModuleType("modules.core")

    def _update_status(message: str, scope: str = "DLC.API") -> None:
        print(f"[{scope}] {message}")

    _core_stub.update_status = _update_status  # type: ignore[attr-defined]
    sys.modules["modules.core"] = _core_stub

from api.config import settings
from api.database import init_db
from api.queue import job_queue
from api.storage import cleanup_old_results


def _configure_globals() -> None:
    """Set modules.globals for headless API mode."""
    import modules.globals

    modules.globals.headless = True
    modules.globals.many_faces = False
    modules.globals.map_faces = False
    modules.globals.mouth_mask = False
    modules.globals.poisson_blend = False

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        modules.globals.execution_providers = [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
    else:
        modules.globals.execution_providers = ["CPUExecutionProvider"]


async def _periodic_cleanup(interval: int = 3600) -> None:
    """Background task: delete old result files periodically."""
    while True:
        await asyncio.sleep(interval)
        try:
            deleted = cleanup_old_results()
            if deleted:
                print(f"[cleanup] Removed {deleted} expired result files")
        except Exception as exc:
            print(f"[cleanup] Error: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init database
    await init_db()

    # Configure ML globals and warm up models
    _configure_globals()
    from modules.face_analyser import get_face_analyser
    from modules.processors.frame.face_swapper import get_face_swapper

    get_face_analyser()
    get_face_swapper()

    # Start job queue worker
    job_queue.start()

    # Start background cleanup task
    cleanup_task = asyncio.create_task(_periodic_cleanup())

    yield

    cleanup_task.cancel()


app = FastAPI(
    title="Deep-Live-Cam API",
    description="Face swap SaaS API powered by Deep-Live-Cam",
    version="2.0.0",
    lifespan=lifespan,
)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key=settings.jwt_secret)


@app.middleware("http")
async def session_cookie_middleware(request: Request, call_next):
    """Ensure anonymous users get a session cookie for usage tracking."""
    response: Response = await call_next(request)
    if "dlc_session" not in request.cookies:
        response.set_cookie(
            "dlc_session",
            uuid.uuid4().hex,
            httponly=True,
            samesite="lax",
            max_age=86400 * 30,
        )
    return response


# --- Mount routers ---
from api.routers import auth, health, jobs, payments, swap, ui, user  # noqa: E402

app.include_router(ui.router)
app.include_router(health.router)
app.include_router(swap.router)
app.include_router(jobs.router)
app.include_router(auth.router)
app.include_router(payments.router)
app.include_router(user.router)
