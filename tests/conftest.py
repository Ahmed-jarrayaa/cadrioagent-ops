"""Configuration commune aux tests du projet.

``audio_engine`` n'est pas un paquet installé : ses modules s'importent
directement (``import registry``, ``from preprocess import ...``). On ajoute donc
son dossier au ``sys.path`` pour les tests de ``tests/test_audio.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIO_ENGINE = ROOT / "audio_engine"

if str(AUDIO_ENGINE) not in sys.path:
    sys.path.insert(0, str(AUDIO_ENGINE))
