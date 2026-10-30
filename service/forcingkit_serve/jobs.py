"""In-process queue for slow store builds (`/api/v1/jobs`).

A job's id is the id of the store it builds, so identical submissions share one build instead of
racing to write the same store. Jobs live in memory: the service runs as one process, and a store
that finished before a restart is found on disk by the route that reads job status.
"""

import concurrent.futures
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from forcingkit import settings

logger = logging.getLogger(__name__)

# Seconds a finished job stays in memory; after that its store, if any, is found on disk.
FINISHED_TTL_S = 24 * 3600

ACTIVE = ("queued", "running")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"
    submitted: str = field(default_factory=_now)
    started: Optional[str] = None
    finished: Optional[str] = None
    error: Optional[str] = None
    # The build's response (zarr_id, zarr_path, donor, ...) once it has succeeded.
    result: Optional[Dict[str, Any]] = None
    finished_at: float = 0.0


class JobRegistry:
    def __init__(self, workers: Optional[int] = None):
        self._workers = workers
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def _pool(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._workers or settings.job_workers(),
                thread_name_prefix="forcingkit-job",
            )
        return self._executor

    def _expire(self) -> None:
        cutoff = time.monotonic() - FINISHED_TTL_S
        for job_id in [
            j.id
            for j in self._jobs.values()
            if j.status not in ACTIVE and j.finished_at < cutoff
        ]:
            del self._jobs[job_id]

    def submit(
        self,
        job_id: str,
        kind: str,
        build: Callable[[], Dict[str, Any]],
        rebuild: bool = False,
    ) -> Job:
        """Queue `build` under `job_id`, or return the job already queued, running or succeeded
        under that id. A failed job, or a succeeded one when `rebuild` is set, is replaced by a
        fresh attempt; an active one is always shared."""
        with self._lock:
            self._expire()
            job = self._jobs.get(job_id)
            if job is not None and (
                job.status in ACTIVE or (job.status == "succeeded" and not rebuild)
            ):
                return job
            job = Job(id=job_id, kind=kind)
            self._jobs[job_id] = job
        self._pool().submit(self._run, job, build)
        return job

    def finished(self, job_id: str, kind: str, result: Dict[str, Any]) -> Job:
        """Record a job that needed no build (its store was already complete)."""
        job = Job(id=job_id, kind=kind, status="succeeded", result=result)
        job.started = job.finished = job.submitted
        job.finished_at = time.monotonic()
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def is_active(self, job_id: str) -> bool:
        job = self.get(job_id)
        return job is not None and job.status in ACTIVE

    def _run(self, job: Job, build: Callable[[], Dict[str, Any]]) -> None:
        job.status, job.started = "running", _now()
        try:
            job.result = build()
            job.status = "succeeded"
        except Exception as e:
            logger.error(f"Job {job.id} ({job.kind}) failed: {e}")
            job.error = str(getattr(e, "detail", None) or e)
            job.status = "failed"
        job.finished, job.finished_at = _now(), time.monotonic()


registry = JobRegistry()
