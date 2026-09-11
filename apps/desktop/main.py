"""Minimal Windows-first desktop shell.

The window appears immediately; hardware detection runs on a worker thread and the
results are marshaled back to the Tk event loop. It remains a thin presentation layer
over ``core`` and has no network dependency.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

from core.editions import resolve_entitlements
from core.hardware.profiler import HardwareProfiler
from core.sizing.engine import FitResult, ModelFitEngine
from schemas.hardware import HardwareProfile


@dataclass
class _Profiled:
    profile: HardwareProfile
    recommendations: list[FitResult]


def _detect(result_queue: queue.Queue[object]) -> None:
    try:
        profile = HardwareProfiler().profile()
        recommendations = ModelFitEngine().recommend(profile)
        result_queue.put(_Profiled(profile, recommendations))
    except Exception as exc:  # surfaced in the UI, not swallowed
        result_queue.put(exc)


def main() -> None:
    """Launch the local hardware recommendation screen."""
    root = tk.Tk()
    root.title("Whittle")
    root.geometry("900x640")
    frame = ttk.Frame(root, padding=24)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame, text="Whittle", font=("Segoe UI", 18, "bold")
    ).pack(anchor="w")
    edition = resolve_entitlements().edition.value
    ttk.Label(frame, text=f"Edition: {edition}").pack(anchor="w", pady=(2, 12))
    status = ttk.Label(frame, text="Detecting hardware...", font=("Segoe UI", 11))
    status.pack(anchor="w")

    result_queue: queue.Queue[object] = queue.Queue()
    threading.Thread(target=_detect, args=(result_queue,), daemon=True).start()

    def poll() -> None:
        try:
            item = result_queue.get_nowait()
        except queue.Empty:
            root.after(100, poll)
            return
        status.destroy()
        if isinstance(item, _Profiled):
            _render(frame, item)
        else:
            _render_error(frame, item)

    root.after(100, poll)
    root.mainloop()


def _render(frame: ttk.Frame, data: _Profiled) -> None:
    profile = data.profile
    device = (
        f"{profile.os} | {profile.architecture} | {profile.cpu.model} | "
        f"{profile.memory.total_gb:.1f} GB RAM ({profile.memory.available_gb:.1f} GB available)"
    )
    ttk.Label(frame, text=device, wraplength=820).pack(anchor="w", pady=(4, 6))
    gpu_text = (
        f"GPU: {profile.gpu.vendor} {profile.gpu.model} ({profile.gpu.memory_gb:.1f} GB)"
    )
    ttk.Label(frame, text=gpu_text).pack(anchor="w", pady=(0, 18))
    ttk.Label(
        frame, text="Hardware Recommendation", font=("Segoe UI", 13, "bold")
    ).pack(anchor="w")
    for result in data.recommendations:
        title = f"{result.model_size}  -  {result.status.replace('_', ' ').title()}"
        card = ttk.LabelFrame(frame, text=title, padding=10)
        card.pack(fill="x", pady=6)
        summary = (
            f"Memory: {result.memory_required_gb:.1f} GB | "
            f"Disk: {result.disk_required_gb:.1f} GB | "
            f"Confidence: {result.confidence} | "
            f"Time: {result.estimated_training_time.min_hours:.1f}-"
            f"{result.estimated_training_time.max_hours:.1f} hours"
        )
        ttk.Label(card, text=summary).pack(anchor="w")
        ttk.Label(card, text="; ".join(result.reasons[:3]), wraplength=820).pack(anchor="w")


def _render_error(frame: ttk.Frame, error: object) -> None:
    ttk.Label(
        frame, text="Hardware detection failed", font=("Segoe UI", 13, "bold"), foreground="#b00020"
    ).pack(anchor="w", pady=(8, 4))
    ttk.Label(frame, text=str(error), wraplength=820).pack(anchor="w")


if __name__ == "__main__":
    main()
