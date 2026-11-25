"""Pipeline RAG CardioAgent-Ops.

Les variables ``USE_TF`` et ``USE_FLAX`` sont désactivées avant tout import de
``transformers``/``sentence-transformers`` : le dépôt utilise le backend PyTorch uniquement.
"""

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
