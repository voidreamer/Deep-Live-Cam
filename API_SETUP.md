# Deep-Live-Cam REST API

Face-swap REST API and web UI powered by Deep-Live-Cam. Upload a source face and a target image or video, get the swapped result back.

## Quick Start

```bash
# 1. Clone and set up
git clone <repo-url> && cd Deep-Live-Cam
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Download the face-swap model (required, ~529 MB)
curl -L -o models/inswapper_128.onnx \
  "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx"

# 3. Run
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1

# 4. Open http://localhost:8000
```

## Models

| Model | File | Location | Required |
|-------|------|----------|----------|
| Face analyser | `buffalo_l` | `~/.insightface/models/` (auto-downloads on first run) | Yes |
| Face swapper | `inswapper_128.onnx` | `models/` | Yes |
| Face enhancer | `GFPGANv1.4.pth` | `models/` | No (only for `?enhance=true`) |

**Important:** Use `inswapper_128.onnx` (not `_fp16` variant, which is CUDA-only).

## Web UI

The root `/` serves a single-page web app with:

- **Image tab** -- upload source face + target image, get swapped JPEG
- **Video tab** -- upload source face + target video, get swapped MP4 with live progress bar
- **Options** -- "Swap all faces" and "Enhance" toggles
- Dark theme, mobile-friendly

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Model status, execution providers |
| `GET` | `/docs` | Swagger UI (auto-generated) |
| `POST` | `/swap` | Image face swap (returns JPEG) |
| `POST` | `/swap/video` | Submit video swap job (returns `{job_id}`) |
| `GET` | `/job/{id}` | Poll video job progress |
| `GET` | `/job/{id}/download` | Download completed video result |

### POST /swap

Multipart form with `source` (image) and `target` (image). Query params:

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `many_faces` | bool | `false` | Swap all detected faces in target |
| `enhance` | bool | `false` | Apply GFPGAN face enhancement |

Returns `image/jpeg`.

```bash
curl -X POST http://localhost:8000/swap \
  -F "source=@source.jpg" \
  -F "target=@target.jpg" \
  --output result.jpg
```

### POST /swap/video

Same params as `/swap` but `target` is a video file (up to 100 MB). Returns `{job_id}` immediately. Processing runs in a background thread.

```bash
# Submit
curl -X POST http://localhost:8000/swap/video \
  -F "source=@face.jpg" -F "target=@video.mp4" | jq .

# Poll progress
curl http://localhost:8000/job/<job_id>

# Download when done
curl http://localhost:8000/job/<job_id>/download --output result.mp4
```

### GET /health

```json
{
  "status": "ok",
  "execution_providers": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
  "models": {
    "face_analyser": {"loaded": true},
    "face_swapper": {"loaded": true, "model_exists": true},
    "face_enhancer": {"model_exists": false}
  }
}
```

`status` is `"ok"` when all required models are loaded, `"degraded"` otherwise.

### Error Codes

| Code | Meaning |
|------|---------|
| 400 | Image can't be decoded, no face detected, or file too large |
| 404 | Job ID not found |
| 500 | Model failure or encoding error |

### Upload Limits

- Images: 10 MB per file
- Videos: 100 MB per file

## Architecture

### Key Design Decisions

1. **Stubbed `modules.core`** -- The existing codebase imports `tensorflow` at module level via `modules/core.py`. The API injects a lightweight stub into `sys.modules` before importing face processing modules, avoiding a ~2 GB dependency. Only `update_status()` (a print wrapper) is needed.

2. **Apple Silicon detection** -- Auto-configures `CoreMLExecutionProvider` + `CPUExecutionProvider` on ARM64 Darwin. Falls back to CPU-only if CoreML options aren't supported by the installed onnxruntime.

3. **Single worker** -- Models are singletons. Multiple workers would duplicate GPU/ANE memory.

4. **Job-based video processing** -- Videos are processed in a background thread to avoid HTTP timeout issues (Cloudflare tunnels have a 100s default). The frontend polls `/job/{id}` every second and renders a live progress bar.

5. **CORS enabled** -- `allow_origins=["*"]` so the API can be called from any frontend.

### File Structure

```
api.py              -- FastAPI server (single file: UI + API + job runner)
models/
  inswapper_128.onnx   -- face swap model (gitignored)
  GFPGANv1.4.pth       -- face enhancer model (optional, gitignored)
modules/
  globals.py            -- global config state
  face_analyser.py      -- face detection (insightface buffalo_l)
  processors/frame/
    face_swapper.py     -- face swap (insightface inswapper)
    face_enhancer.py    -- face enhancement (GFPGAN)
```

## Deployment

### Local Network

The server binds to `0.0.0.0:8000`, accessible from any device on the same network:
```
http://<your-local-ip>:8000
```

### Cloudflare Tunnel

To expose externally, add an ingress entry in `~/.cloudflared/config.yml`:

```yaml
ingress:
  - hostname: dlc.yourdomain.com
    service: http://127.0.0.1:8000
  # ... other services ...
  - service: http_status:404
```

Then add the DNS route and restart:
```bash
cloudflared tunnel route dns <tunnel-id> dlc.yourdomain.com
kill $(pgrep -f 'cloudflared tunnel run')
cloudflared tunnel run &
```

### Production Notes

- Use `--workers 1` always (model singletons)
- Video processing is CPU-bound (~1-3 sec/frame on CPU). Keep videos short or use a machine with CUDA.
- Temp files from video jobs are cleaned up after download or on server shutdown
- The `buffalo_l` face detection model auto-downloads on first startup (~275 MB)
