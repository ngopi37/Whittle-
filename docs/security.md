# Security

SG2 makes no network calls and executes no dataset, model, or license content.

## Current guarantees

- **Subprocesses** use argument arrays (never a shell). The hardware profiler runs
  one constant PowerShell script, resolved to the in-box
  `System32\WindowsPowerShell\v1.0\powershell.exe` (falling back to `powershell` /
  `pwsh`), with `-NoProfile -NonInteractive` and a timeout.
- **Registry** access (GPU memory) is read-only via stdlib `winreg`, wrapped so any
  failure degrades to "unknown".
- **Dataset ingest** treats all imported content as data: JSONL is parsed with
  `json.loads` and only a `text` field is kept; nothing is `eval`/`exec`'d. Each
  source file's path, size, mtime, and line count are recorded as local provenance.
- **Paths**: user-controlled paths written under a run directory go through
  `core.safety.paths.safe_resolve`, which rejects `..` traversal and absolute
  escapes.
- **Model profiles** and any YAML are loaded with `yaml.safe_load`.
- **License files** are read locally only, never transmitted over the network, and
  cryptographically verified (`core.licensing`) before a paid edition is granted.

## Required before the relevant phase

- **Archive extraction** (`safe_extract`) must validate every member with
  `safe_resolve`, refuse absolute and symlink members, and never execute content.
- **Training workers** run out-of-process over `JobSpec` v1 and must not be handed
  ambient credentials.
- **Registry / deploy** (paid) must authenticate and sign packages.

Report security issues privately to the maintainers. Do not include sensitive
datasets or credentials in reports.
