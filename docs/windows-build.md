# Windows Build

The pipeline stages (P2–P5, GGUF target) have landed; the PyInstaller build itself
is still not started — this remains a plan, not a report of what exists. Target: a
**PyInstaller one-folder** build, produced **on Windows** with the target Python
version — a Linux build cannot produce a valid Windows executable. `torch` and
`llama-cpp-python`'s compiled extensions make the frozen build considerably larger
than P1's; budget time for that in whatever ships this first.

## Planned build notes

- **Two entry points:** `apps/cli/main.py` (`sg2-model`) and `apps/desktop/main.py`.
  A one-folder build keeps startup fast and lets the model catalog / configs ship as
  data files.
- **Data files:** bundle `configs/models/*.yaml` (`--add-data`) — `core.catalog`
  resolves them relative to the package root.
- **Tkinter:** the desktop shell needs the Tcl/Tk runtime. Verify `tcl86t.dll` /
  `tk86t.dll` and the `tcl/` `tk/` data directories are collected; add
  `--collect-all tkinter` if PyInstaller misses them on the build host.
- **Pydantic v2:** `--collect-all pydantic` and `--collect-all pydantic_core` (the
  compiled `pydantic_core` extension is easy to miss).
- **psutil:** ships a compiled extension; confirm it loads from the frozen build.
- **PowerShell is an OS dependency, not bundled.** The profiler shells out to
  `powershell.exe`; it is present on all supported Windows. No action needed, but do
  not attempt to bundle it.
- **No hidden network deps:** keep the build offline; there is nothing to fetch at
  runtime.
- **Code signing:** sign both the folder's executables and, later, the model
  packages. Unsigned builds trigger SmartScreen.

## Acceptance

The frozen CLI runs `hardware`, `recommend`, `project`, and `pipeline plan` with no
Python install present; the frozen desktop opens, detects hardware on its worker
thread, and renders recommendations.
