import io
import os
import platform
import sys
import tempfile
import threading
import time
import types
import uuid
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

# Stub modules.core so face_swapper/face_enhancer don't pull in tensorflow
if "modules.core" not in sys.modules:
    _core_stub = types.ModuleType("modules.core")

    def _update_status(message: str, scope: str = "DLC.API") -> None:
        print(f"[{scope}] {message}")

    _core_stub.update_status = _update_status  # type: ignore[attr-defined]
    sys.modules["modules.core"] = _core_stub

import modules.globals  # noqa: E402
from modules.face_analyser import get_face_analyser, get_many_faces, get_one_face  # noqa: E402
from modules.processors.frame.face_swapper import get_face_swapper, swap_face  # noqa: E402

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB per image
MAX_VIDEO_BYTES = 100 * 1024 * 1024  # 100 MB per video

# --- Job store for async video processing ---
jobs: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
JOB_TTL = 600  # seconds to keep completed jobs before cleanup


def _configure_globals() -> None:
    """Set modules.globals for headless API mode."""
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_globals()
    get_face_analyser()
    get_face_swapper()
    yield
    # Cleanup temp files from any remaining jobs
    with JOBS_LOCK:
        for job in jobs.values():
            for path in (job.get("tmp_in"), job.get("tmp_out")):
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass


app = FastAPI(
    title="Deep-Live-Cam API",
    description="Face swap REST API powered by Deep-Live-Cam",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Deep-Live-Cam</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #0f0f0f; color: #e0e0e0; min-height: 100vh;
         display: flex; flex-direction: column; align-items: center; padding: 2rem 1rem; }
  h1 { font-size: 1.5rem; margin-bottom: 1.5rem; color: #fff; }
  .container { width: 100%; max-width: 900px; }
  .tabs { display: flex; gap: 0; margin-bottom: 1rem; }
  .tab { padding: .5rem 1.25rem; background: #1a1a1a; border: 1px solid #333; cursor: pointer;
         font-size: .85rem; color: #888; transition: all .2s; }
  .tab:first-child { border-radius: 8px 0 0 8px; }
  .tab:last-child { border-radius: 0 8px 8px 0; }
  .tab.active { background: #4a9eff; color: #fff; border-color: #4a9eff; }
  .upload-row { display: flex; gap: 1rem; margin-bottom: 1rem; }
  .upload-box { flex: 1; border: 2px dashed #333; border-radius: 12px; padding: 1rem;
                text-align: center; cursor: pointer; transition: border-color .2s;
                position: relative; overflow: hidden; min-height: 200px;
                display: flex; flex-direction: column; align-items: center; justify-content: center; }
  .upload-box:hover { border-color: #666; }
  .upload-box.has-image { border-color: #4a9eff; }
  .upload-box input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
  .upload-box img { max-width: 100%; max-height: 250px; border-radius: 8px; }
  .upload-box .label { font-size: .85rem; color: #888; margin-top: .5rem; }
  .upload-box .placeholder { color: #555; font-size: 2rem; margin-bottom: .25rem; }
  .options { display: flex; gap: 1.5rem; margin-bottom: 1rem; align-items: center; }
  .options label { display: flex; align-items: center; gap: .4rem; font-size: .9rem; cursor: pointer; }
  .options input[type=checkbox] { accent-color: #4a9eff; width: 16px; height: 16px; }
  button { background: #4a9eff; color: #fff; border: none; border-radius: 8px;
           padding: .75rem 2rem; font-size: 1rem; cursor: pointer; font-weight: 600;
           transition: background .2s; }
  button:hover { background: #3a8eef; }
  button:disabled { background: #333; color: #666; cursor: not-allowed; }
  .result-area { margin-top: 1.5rem; text-align: center; }
  .result-area img { max-width: 100%; border-radius: 12px; border: 1px solid #222; }
  .error { color: #ff6b6b; margin-top: 1rem; font-size: .9rem; }
  .spinner { display: none; margin: 1rem auto; width: 32px; height: 32px;
             border: 3px solid #333; border-top-color: #4a9eff; border-radius: 50%;
             animation: spin .8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .status { font-size: .75rem; color: #555; margin-top: 2rem; }
  .download-btn { display: inline-block; margin-top: .75rem; background: #2a2a2a;
                  color: #4a9eff; padding: .5rem 1.25rem; border-radius: 6px;
                  text-decoration: none; font-size: .85rem; border: 1px solid #333; }
  .download-btn:hover { background: #333; }
  .progress-bar-wrap { width: 100%; background: #1a1a1a; border-radius: 8px; height: 24px;
                       margin-top: .75rem; overflow: hidden; display: none; border: 1px solid #333; }
  .progress-bar { height: 100%; background: #4a9eff; border-radius: 8px; transition: width .3s;
                  display: flex; align-items: center; justify-content: center;
                  font-size: .75rem; color: #fff; font-weight: 600; min-width: 2rem; }
  .progress-text { font-size: .8rem; color: #888; margin-top: .4rem; }
</style>
</head>
<body>
<h1>Deep-Live-Cam</h1>
<div class="container">
  <div class="tabs">
    <div class="tab active" data-mode="image">Image</div>
    <div class="tab" data-mode="video">Video</div>
  </div>
  <div class="upload-row">
    <div class="upload-box" id="source-box">
      <div class="placeholder">+</div>
      <div class="label">Source face</div>
      <input type="file" accept="image/*" id="source-input">
    </div>
    <div class="upload-box" id="target-box">
      <div class="placeholder">+</div>
      <div class="label" id="target-label">Target image</div>
      <input type="file" accept="image/*" id="target-input">
    </div>
  </div>
  <div class="options">
    <label><input type="checkbox" id="many-faces"> Swap all faces</label>
    <label><input type="checkbox" id="enhance"> Enhance</label>
    <button id="swap-btn" disabled>Swap</button>
  </div>
  <div class="spinner" id="spinner"></div>
  <div class="progress-bar-wrap" id="progress-bar-wrap">
    <div class="progress-bar" id="progress-bar">0%</div>
  </div>
  <div class="progress-text" id="progress-text"></div>
  <div class="error" id="error"></div>
  <div class="result-area" id="result-area"></div>
  <div class="status" id="status"></div>
</div>
<script>
const sourceInput = document.getElementById('source-input');
const targetInput = document.getElementById('target-input');
const sourceBox = document.getElementById('source-box');
const targetBox = document.getElementById('target-box');
const targetLabel = document.getElementById('target-label');
const swapBtn = document.getElementById('swap-btn');
const spinner = document.getElementById('spinner');
const progressWrap = document.getElementById('progress-bar-wrap');
const progressBar = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-text');
const errorEl = document.getElementById('error');
const resultArea = document.getElementById('result-area');
const statusEl = document.getElementById('status');
let sourceFile = null, targetFile = null, mode = 'image';

document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    mode = tab.dataset.mode;
    targetLabel.textContent = mode === 'video' ? 'Target video' : 'Target image';
    targetInput.accept = mode === 'video' ? 'video/*' : 'image/*';
    targetFile = null;
    const existingMedia = targetBox.querySelector('img, video');
    if (existingMedia) existingMedia.remove();
    const ph = targetBox.querySelector('.placeholder');
    if (!ph) { const d = document.createElement('div'); d.className='placeholder'; d.textContent='+'; targetBox.insertBefore(d, targetBox.firstChild); }
    targetBox.classList.remove('has-image');
    targetInput.value = '';
    updateBtn();
  });
});

function preview(input, box, isVideo) {
  const file = input.files[0];
  if (!file) return null;
  const existingMedia = box.querySelector('img, video');
  const placeholder = box.querySelector('.placeholder');
  const label = box.querySelector('.label');
  if (existingMedia) existingMedia.remove();
  if (placeholder) placeholder.remove();
  if (isVideo) {
    const vid = document.createElement('video');
    vid.src = URL.createObjectURL(file);
    vid.style.maxWidth = '100%'; vid.style.maxHeight = '250px'; vid.style.borderRadius = '8px';
    vid.muted = true; vid.loop = true; vid.autoplay = true;
    box.insertBefore(vid, box.firstChild);
  } else {
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    box.insertBefore(img, box.firstChild);
  }
  if (label) label.textContent = file.name;
  box.classList.add('has-image');
  return file;
}

sourceInput.addEventListener('change', () => { sourceFile = preview(sourceInput, sourceBox, false); updateBtn(); });
targetInput.addEventListener('change', () => { targetFile = preview(targetInput, targetBox, mode === 'video'); updateBtn(); });
function updateBtn() { swapBtn.disabled = !(sourceFile && targetFile); }

function resetUI() {
  spinner.style.display = 'none';
  progressWrap.style.display = 'none';
  progressText.textContent = '';
  progressBar.style.width = '0%';
  progressBar.textContent = '0%';
  updateBtn();
}

// --- Image swap (direct) ---
async function doImageSwap(params, fd) {
  spinner.style.display = 'block';
  const resp = await fetch('/swap?' + params.toString(), { method: 'POST', body: fd });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || 'Swap failed');
  }
  const blob = await resp.blob();
  const url = URL.createObjectURL(blob);
  const img = document.createElement('img'); img.src = url;
  resultArea.appendChild(img);
  const dl = document.createElement('a');
  dl.href = url; dl.download = 'result.jpg'; dl.className = 'download-btn'; dl.textContent = 'Download';
  resultArea.appendChild(dl);
}

// --- Video swap (job-based with progress) ---
async function doVideoSwap(params, fd) {
  // 1. Submit job
  progressText.textContent = 'Uploading video...';
  spinner.style.display = 'block';
  const submitResp = await fetch('/swap/video?' + params.toString(), { method: 'POST', body: fd });
  if (!submitResp.ok) {
    const err = await submitResp.json().catch(() => ({ detail: submitResp.statusText }));
    throw new Error(err.detail || 'Failed to submit video job');
  }
  const { job_id } = await submitResp.json();
  spinner.style.display = 'none';
  progressWrap.style.display = 'block';
  progressText.textContent = 'Processing frames...';

  // 2. Poll for progress
  while (true) {
    await new Promise(r => setTimeout(r, 1000));
    const pollResp = await fetch('/job/' + job_id);
    if (!pollResp.ok) throw new Error('Failed to check job status');
    const job = await pollResp.json();

    if (job.status === 'processing') {
      const pct = job.total_frames > 0 ? Math.round((job.processed_frames / job.total_frames) * 100) : 0;
      progressBar.style.width = pct + '%';
      progressBar.textContent = pct + '%';
      progressText.textContent = 'Frame ' + job.processed_frames + ' / ' + job.total_frames;
    } else if (job.status === 'done') {
      progressBar.style.width = '100%';
      progressBar.textContent = '100%';
      progressText.textContent = 'Done! Downloading result...';
      // 3. Download result
      const dlResp = await fetch('/job/' + job_id + '/download');
      if (!dlResp.ok) throw new Error('Failed to download result');
      const blob = await dlResp.blob();
      const url = URL.createObjectURL(blob);
      const vid = document.createElement('video');
      vid.src = url; vid.controls = true; vid.autoplay = true;
      vid.style.maxWidth = '100%'; vid.style.borderRadius = '12px'; vid.style.border = '1px solid #222';
      resultArea.appendChild(vid);
      const dl = document.createElement('a');
      dl.href = url; dl.download = 'result.mp4'; dl.className = 'download-btn'; dl.textContent = 'Download MP4';
      resultArea.appendChild(dl);
      break;
    } else if (job.status === 'failed') {
      throw new Error(job.error || 'Video processing failed');
    }
  }
}

swapBtn.addEventListener('click', async () => {
  errorEl.textContent = '';
  resultArea.innerHTML = '';
  resetUI();
  swapBtn.disabled = true;

  const params = new URLSearchParams();
  if (document.getElementById('many-faces').checked) params.set('many_faces', 'true');
  if (document.getElementById('enhance').checked) params.set('enhance', 'true');
  const fd = new FormData();
  fd.append('source', sourceFile);
  fd.append('target', targetFile);

  try {
    if (mode === 'video') {
      await doVideoSwap(params, fd);
    } else {
      await doImageSwap(params, fd);
    }
  } catch (e) {
    errorEl.textContent = e.message;
  } finally {
    resetUI();
  }
});

fetch('/health').then(r => r.json()).then(d => {
  const s = d.status === 'ok' ? 'All models loaded' : 'Degraded -- check /health for details';
  statusEl.textContent = s + ' | ' + d.execution_providers.join(', ');
}).catch(() => { statusEl.textContent = 'Could not reach API'; });
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return INDEX_HTML


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_image(data: bytes, label: str) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail=f"Could not decode {label} image")
    return img


def _process_video_job(job_id: str, source_face, tmp_in: str, tmp_out: str,
                       many_faces: bool, enhance: bool) -> None:
    """Run video face-swap in a background thread, updating job progress."""
    try:
        cap = cv2.VideoCapture(tmp_in)
        if not cap.isOpened():
            with JOBS_LOCK:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = "Could not open target video"
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        with JOBS_LOCK:
            jobs[job_id]["total_frames"] = total

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_out, fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            with JOBS_LOCK:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = "Failed to create output video writer"
            return

        enhance_fn = None
        if enhance:
            try:
                from modules.processors.frame.face_enhancer import enhance_face
                enhance_fn = enhance_face
            except Exception:
                pass

        processed = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if many_faces:
                faces = get_many_faces(frame)
            else:
                one = get_one_face(frame)
                faces = [one] if one else None

            if faces:
                for tf in faces:
                    frame = swap_face(source_face, tf, frame)

            if enhance_fn is not None:
                frame = enhance_fn(frame)

            writer.write(frame)
            processed += 1

            with JOBS_LOCK:
                jobs[job_id]["processed_frames"] = processed

        cap.release()
        writer.release()

        if processed == 0:
            with JOBS_LOCK:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = "Video contained no readable frames"
            return

        with JOBS_LOCK:
            jobs[job_id]["processed_frames"] = processed
            jobs[job_id]["status"] = "done"
            jobs[job_id]["finished_at"] = time.time()

    except Exception as exc:
        with JOBS_LOCK:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    swapper_loaded = get_face_swapper() is not None
    analyser_loaded = get_face_analyser() is not None
    models_dir = os.path.join(os.path.dirname(__file__), "models")
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


@app.post("/swap")
async def swap(
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    many_faces: bool = Query(False),
    enhance: bool = Query(False),
):
    source_bytes = await source.read()
    if len(source_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Source file exceeds 10 MB limit")

    target_bytes = await target.read()
    if len(target_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Target file exceeds 10 MB limit")

    source_img = _decode_image(source_bytes, "source")
    target_img = _decode_image(target_bytes, "target")

    source_face = get_one_face(source_img)
    if source_face is None:
        raise HTTPException(status_code=400, detail="No face detected in source image")

    if many_faces:
        target_faces = get_many_faces(target_img)
        if not target_faces:
            raise HTTPException(status_code=400, detail="No faces detected in target image")
    else:
        one = get_one_face(target_img)
        if one is None:
            raise HTTPException(status_code=400, detail="No face detected in target image")
        target_faces = [one]

    try:
        result = target_img
        for tf in target_faces:
            result = swap_face(source_face, tf, result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Face swap failed: {exc}")

    if enhance:
        try:
            from modules.processors.frame.face_enhancer import enhance_face
            result = enhance_face(result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Face enhancement failed: {exc}")

    ok, buf = cv2.imencode(".jpg", result)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode result image")

    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")


@app.post("/swap/video")
async def swap_video(
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    many_faces: bool = Query(False),
    enhance: bool = Query(False),
):
    source_bytes = await source.read()
    if len(source_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Source file exceeds 10 MB limit")

    target_bytes = await target.read()
    if len(target_bytes) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=400, detail="Target video exceeds 100 MB limit")

    source_img = _decode_image(source_bytes, "source")
    source_face = get_one_face(source_img)
    if source_face is None:
        raise HTTPException(status_code=400, detail="No face detected in source image")

    suffix = os.path.splitext(target.filename or "video.mp4")[1] or ".mp4"
    tmp_in = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_in.write(target_bytes)
    tmp_in.close()
    tmp_out.close()

    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        jobs[job_id] = {
            "status": "processing",
            "total_frames": 0,
            "processed_frames": 0,
            "error": None,
            "tmp_in": tmp_in.name,
            "tmp_out": tmp_out.name,
            "created_at": time.time(),
            "finished_at": None,
        }

    thread = threading.Thread(
        target=_process_video_job,
        args=(job_id, source_face, tmp_in.name, tmp_out.name, many_faces, enhance),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@app.get("/job/{job_id}")
async def job_status(job_id: str):
    with JOBS_LOCK:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "status": job["status"],
        "total_frames": job["total_frames"],
        "processed_frames": job["processed_frames"],
        "error": job["error"],
    }


@app.get("/job/{job_id}/download")
async def job_download(job_id: str):
    with JOBS_LOCK:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Job is not done (status: {job['status']})")

    tmp_out = job["tmp_out"]
    if not os.path.exists(tmp_out):
        raise HTTPException(status_code=500, detail="Result file missing")

    with open(tmp_out, "rb") as f:
        video_bytes = f.read()

    # Cleanup temp files after download
    for path in (job.get("tmp_in"), job.get("tmp_out")):
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass
    with JOBS_LOCK:
        jobs.pop(job_id, None)

    return StreamingResponse(
        io.BytesIO(video_bytes),
        media_type="video/mp4",
        headers={"Content-Disposition": "attachment; filename=result.mp4"},
    )
