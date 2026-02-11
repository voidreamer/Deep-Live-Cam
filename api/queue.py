"""Priority queue-based video job runner."""

from __future__ import annotations

import os
import queue
import threading
import time
from datetime import datetime, timezone

import cv2

from api.config import settings
from api.storage import result_path


class _JobItem:
    """Wrapper for priority queue ordering."""

    __slots__ = ("priority", "seq", "job_id", "payload")

    def __init__(self, priority: int, seq: int, job_id: str, payload: dict):
        self.priority = priority
        self.seq = seq
        self.job_id = job_id
        self.payload = payload

    def __lt__(self, other: _JobItem) -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.seq < other.seq


class JobQueue:
    """Single-worker priority queue for video processing jobs.

    Priority 0 = premium (processed first), 1 = free.
    """

    def __init__(self) -> None:
        self._q: queue.PriorityQueue[_JobItem] = queue.PriorityQueue()
        self._seq = 0
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        # In-memory job state for progress tracking (mirrors DB but avoids async)
        self._job_state: dict[str, dict] = {}
        self._state_lock = threading.Lock()

    def start(self) -> None:
        """Start the background worker thread."""
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def enqueue(self, job_id: str, priority: int, payload: dict) -> None:
        """Add a job to the queue."""
        with self._lock:
            self._seq += 1
            seq = self._seq
        with self._state_lock:
            self._job_state[job_id] = {
                "status": "queued",
                "total_frames": 0,
                "processed_frames": 0,
                "error": None,
            }
        self._q.put(_JobItem(priority, seq, job_id, payload))

    def get_state(self, job_id: str) -> dict | None:
        """Get current in-memory job state."""
        with self._state_lock:
            return self._job_state.get(job_id, {}).copy() or None

    def _update(self, job_id: str, **kw) -> None:
        with self._state_lock:
            if job_id in self._job_state:
                self._job_state[job_id].update(kw)

    def _run(self) -> None:
        """Worker loop: pull jobs from queue and process them."""
        while True:
            try:
                item = self._q.get(timeout=5)
            except queue.Empty:
                continue
            try:
                self._process(item)
            except Exception as exc:
                self._update(item.job_id, status="failed", error=str(exc))
            finally:
                self._q.task_done()

    def _process(self, item: _JobItem) -> None:
        job_id = item.job_id
        p = item.payload
        self._update(job_id, status="processing")

        source_face = p["source_face"]
        tmp_in = p["tmp_in"]
        many_faces = p.get("many_faces", False)
        enhance = p.get("enhance", False)

        # Lazy imports to avoid circular deps
        from modules.face_analyser import get_many_faces, get_one_face
        from modules.processors.frame.face_swapper import swap_face

        cap = cv2.VideoCapture(tmp_in)
        if not cap.isOpened():
            self._update(job_id, status="failed", error="Could not open target video")
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._update(job_id, total_frames=total)

        out_path = result_path(job_id, ".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            self._update(job_id, status="failed", error="Failed to create output video writer")
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
            self._update(job_id, processed_frames=processed)

        cap.release()
        writer.release()

        # Clean up temp input file
        try:
            os.unlink(tmp_in)
        except OSError:
            pass

        if processed == 0:
            self._update(job_id, status="failed", error="Video contained no readable frames")
            return

        self._update(job_id, status="done", processed_frames=processed,
                     result_path=out_path)


# Module-level singleton
job_queue = JobQueue()
