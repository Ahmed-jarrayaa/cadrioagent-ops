"""Implémentation de la phase LLM Fine-Tuning.

On désactive les intégrations TensorFlow/Flax de Transformers avant tout import :
- elles ne sont pas utilisées par ce module (PyTorch uniquement) ;
- leur chargement est fragile sur Windows et ralentit l'initialisation.
"""

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
# Réduit la fragmentation mémoire sur les petits GPU (Windows ne supporte pas
# expandable_segments ni expandable_segments:True).
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8,max_split_size_mb:128"
)
