# Datasets

## Implemented — `data-ingest`

`core/pipeline/stages/data_ingest.py` normalizes local `.jsonl` / `.txt` /
`.parquet` sources into deduped, optionally split `dataset.jsonl` artifact(s) of
`{"text": ...}` records:

- content is **data only** — JSONL/Parquet is parsed, never executed;
- each JSONL line must be a string or an object with a string `text` field;
  Parquet sources must have a `text` string column, read batch-by-batch via
  `pyarrow.parquet.ParquetFile.iter_batches` (never loaded fully into memory);
  blank lines/empty strings are skipped; over-long lines are rejected;
- **dedup** (`--no-dedup` to disable): exact-duplicate `text` values are dropped
  across all sources, in memory, by a `set` of seen strings — fine at this
  project's scale (small on-device corpora, not web-scale);
- **val split** (`--val-ratio`): a deterministic hash of each document's text
  (salted by `--split-seed` if given) assigns it to train or val, independent of
  source file order — the same input always produces the same split. Produces a
  second `dataset` artifact tagged `role: "val"` in its provenance; the primary
  artifact is tagged `role: "train"` (or `"all"` when no split was requested).
  `core.pipeline.stages._dataset.select_dataset_artifact` is how every downstream
  stage (`tokenizer-train`, `pretrain`, `evaluate`) picks the right one.
- per-source provenance (path, bytes, mtime, line count, format) plus
  `duplicates_removed` and split counts, written to `dataset.meta.json` and onto
  the artifact's `provenance`;
- no network access.

```powershell
python -m apps.cli.main pipeline run project.json \
  --stage data-ingest --input corpus.jsonl --input notes.txt --val-ratio 0.1
```

## Still deferred

- Archive ingest — must go through `core.safety.paths.safe_extract` (path-traversal
  safe, no symlink members, no execution). Not part of P2's exit criteria.
- Dataset cards / license capture.
