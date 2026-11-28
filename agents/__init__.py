"""Module Multi-Agents CardioAgent-Ops (LangGraph + guardrails Pydantic).

Les variables TensorFlow/JAX sont désactivées avant tout import de
``transformers`` : le pipeline RAG utilise uniquement PyTorch.
"""

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
