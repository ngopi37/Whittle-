"""Out-of-process workers for long-running stages (``pretrain``, ``quantize``).

Modules here are invoked as ``python -m core.pipeline.workers.<name> <jobspec.json>``
by the corresponding stage, never imported by ``apps.cli`` / ``apps.desktop``.
"""
