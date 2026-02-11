"""Health check endpoint."""

from __future__ import annotations

import os

from fastapi import APIRouter

import modules.globals

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    from modules.face_analyser import get_face_analyser
    from modules.processors.frame.face_swapper import get_face_swapper

    swapper_loaded = get_face_swapper() is not None
    analyser_loaded = get_face_analyser() is not None
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    swapper_model = os.path.join(models_dir, "inswapper_128.onnx")
    enhancer_model = os.path.join(models_dir, "GFPGANv1.4.pth")

    return {
        "status": "ok" if (swapper_loaded and analyser_loaded) else "degraded",
        "execution_providers": modules.globals.execution_providers,
        "models": {
            "face_analyser": {"loaded": analyser_loaded},
            "face_swapper": {
                "loaded": swapper_loaded,
                "model_exists": os.path.exists(swapper_model),
            },
            "face_enhancer": {
                "model_exists": os.path.exists(enhancer_model),
            },
        },
    }
