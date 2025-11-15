from __future__ import annotations

from typing import Any

__all__ = ["app", "main"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .main import app as _app, main as _main

        return {"app": _app, "main": _main}[name]
    raise AttributeError(name)
