"""Implémentation de la phase LLM Fine-Tuning.

On désactive les intégrations TensorFlow/Flax de Transformers avant tout import :
- elles ne sont pas utilisées par ce module (PyTorch uniquement) ;
- leur chargement est fragile sur Windows et ralentit l'initialisation.
"""

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
