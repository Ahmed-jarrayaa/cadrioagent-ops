"""CLI : applique les trois modèles du TP3 (SVM, arbre de décision, CNN) à un WAV.

Point d'entrée conservé pour compatibilité ; toute la logique est dans ``registry.py``.

Usage :
    python analyze_signal.py chemin/vers.wav
    python analyze_signal.py chemin/vers.wav --json
    python analyze_signal.py chemin/vers.wav --families svm,tree
"""

from registry import main

if __name__ == "__main__":
    main()
