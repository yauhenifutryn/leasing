from __future__ import annotations

import time
from dataclasses import dataclass
from collections import OrderedDict
from typing import Any


@dataclass
class CacheItem:
    value: Any
    expires_at: float


class TTLCache:
    def __init__(self, ttl_sec: int = 60) -> None:
        self.ttl_sec = ttl_sec
        self._store: dict[str, CacheItem] = {}

    def get(self, key: str) -> Any | None:
        item = self._store.get(key)
        if not item:
            return None
        if item.expires_at < time.time():
            self._store.pop(key, None)
            return None
        return item.value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = CacheItem(value=value, expires_at=time.time() + self.ttl_sec)

    def clear(self) -> None:
        self._store.clear()


class LRUCache:
    def __init__(self, max_size: int = 256) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        self.max_size = max_size
        self._store: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Any | None:
        if key not in self._store:
            return None
        value = self._store.pop(key)
        self._store[key] = value
        return value

    def set(self, key: str, value: Any) -> None:
        if key in self._store:
            self._store.pop(key)
        self._store[key] = value
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()
