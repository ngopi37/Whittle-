"""A hard block on outbound networking, for tests that must prove no network call
happens — not just assert it in docs.

Patches the actual socket-level primitives (``connect``, ``connect_ex``,
``getaddrinfo``), not a specific HTTP client class, so it catches any library's
attempt to reach the network — torch, tokenizers, huggingface_hub, gguf,
llama-cpp-python, or anything else in the dependency tree — not just the ones we
already know about.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager


class NetworkBlockedError(RuntimeError):
    """Raised when code under a :func:`block_network` guard touches the network."""


@contextmanager
def block_network() -> Iterator[None]:
    """Make any outbound connection attempt or DNS resolution raise immediately."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo

    def _blocked_connect(self: socket.socket, address: object) -> None:
        raise NetworkBlockedError(f"blocked outbound connect() to {address!r}")

    def _blocked_connect_ex(self: socket.socket, address: object) -> int:
        raise NetworkBlockedError(f"blocked outbound connect_ex() to {address!r}")

    def _blocked_getaddrinfo(*args: object, **kwargs: object) -> None:
        raise NetworkBlockedError(f"blocked DNS resolution: getaddrinfo{args!r}")

    socket.socket.connect = _blocked_connect  # type: ignore[assignment]
    socket.socket.connect_ex = _blocked_connect_ex  # type: ignore[assignment]
    socket.getaddrinfo = _blocked_getaddrinfo  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]
        socket.getaddrinfo = original_getaddrinfo
