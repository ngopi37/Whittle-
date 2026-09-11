"""Training-time code (model architecture, worker loop).

Nothing here is imported by ``apps.cli`` or ``apps.desktop`` at module load time —
only by ``core.pipeline.stages.model_init`` (lazily, inside ``run()``) and by
``core.pipeline.workers.train_worker`` (a separate process). Importing this package
pulls in ``torch``.
"""
