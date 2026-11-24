"""Implémentation de la phase LLM Fine-Tuning.

On désactive les intégrations TensorFlow/Flax de Transformers avant tout import :
- elles ne sont pas utilisées par ce module (PyTorch uniquement) ;
- leur chargement est fragile sur Windows et ralentit l'initialisation.
"""

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
# Pas de PYTORCH_CUDA_ALLOC_CONF : sur WDDM, garbage_collection_threshold provoque
# des libérations fréquentes et max_split_size_mb casse les grosses allocations.
# La mémoire est gérée explicitement (torch.cuda.empty_cache après éval/pas).
