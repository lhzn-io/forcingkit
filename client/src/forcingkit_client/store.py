"""Remote Zarr stores with a local, chunk-level cache.

The service serves each finished store read-only over HTTP. `CachingHTTPStore` is a local
directory store that, on a miss, fetches the key from the server and keeps it, so a store is
read remotely the first time and from disk afterwards, and only the chunks actually read are ever
transferred. The cache directory is keyed by store id and store version, so a store the server
rebuilt under the same id is fetched afresh.
"""

import asyncio
import os
import threading
from pathlib import Path
from typing import Any, Optional

import httpx
import xarray as xr
from zarr.storage import LocalStore


class CachingHTTPStore(LocalStore):
    def __init__(self, root: str | Path, url: str, timeout_s: float = 120.0):
        os.makedirs(root, exist_ok=True)
        super().__init__(root, read_only=True)
        self.url = url.rstrip("/")
        self._timeout_s = timeout_s
        self._http: Optional[httpx.Client] = None
        self._lock = threading.Lock()

    def _client(self) -> httpx.Client:
        with self._lock:
            if self._http is None:
                self._http = httpx.Client(timeout=self._timeout_s)
            return self._http

    def _fetch(self, key: str) -> bool:
        """Copy `key` from the server into the cache; False when the server has no such key
        (an unwritten chunk, which reads as the fill value)."""
        response = self._client().get(f"{self.url}/{key}")
        if response.status_code == 404:
            return False
        response.raise_for_status()
        path = os.path.join(str(self.root), key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        partial = f"{path}.{os.getpid()}.{threading.get_ident()}.part"
        with open(partial, "wb") as f:
            f.write(response.content)
        os.replace(partial, path)
        return True

    async def get(self, key: str, prototype: Any = None, byte_range: Any = None) -> Any:
        if not os.path.exists(os.path.join(str(self.root), key)):
            if not await asyncio.to_thread(self._fetch, key):
                return None
        return await super().get(key, prototype, byte_range)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, CachingHTTPStore)
            and other.url == self.url
            and str(other.root) == str(self.root)
        )

    def __hash__(self) -> int:
        return hash((self.url, str(self.root)))


def open_remote(
    url: str, cache_dir: str | Path, store_id: str, version: str
) -> xr.Dataset:
    """Open the store at `url` (a `.../stores/<id>.zarr` href) through the local cache."""
    root = Path(cache_dir).expanduser() / "stores" / store_id / version
    return xr.open_zarr(CachingHTTPStore(root, url), consolidated=True, zarr_format=2)
