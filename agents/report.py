"""Construction du rapport à partir des résultats réels du workflow 5 modèles.

Aucune valeur n'est inventée ici : les sections factuelles sont dérivées du JSON
renvoyé par ``audio_engine.registry.analyze``. La conclusion met en regard ces
faits et un passage réellement récupéré dans les deux documents du projet. Le
LLM est limité à une confirmation structurée ; toute instruction hors contexte
est bloquée avant tout appel LLM.
"""

from __future__ import annotations

import json
import re
from typing import Any

MODEL_KEYS = ("svm", "tree", "cnn", "tl", "hybrid")

# Mots qui autorisent une instruction dans le périmètre du rapport. Une demande
# qui n'en contient aucun est bloquée avant tout appel au LLM.
IN_SCOPE_TERMS = {
    "signal", "signaux", "cardiaque", "coeur", "cœur", "pcg", "ecg", "auscultation",
    "bruit", "bruits", "souffle", "systole", "diastole", "b1", "b2", "valve", "valves",
    "rapport", "analyse", "analyser", "resultat", "resultats", "prediction", "predictions",
    "modele", "modeles", "score", "scores", "probabilite", "probabilites", "seuil",
    "seuils", "caracteristique", "caracteristiques", "classification", "classe",
    "consensus", "vote", "votes", "divergence", "divergences", "document", "documents",
    "source", "sources", "conclusion", "explication", "expliquer", "detail", "detalles",
    "etat", "evaluer", "evaluation", "depistage", "normal", "anormal", "anomalie",
    "abnormal", "heart", "sound", "sounds", "model", "models", "report", "result",
    "results", "feature", "features", "analysis", "classify", "screening",
}

OUT_OF_SCOPE_MARKERS = re.compile(
    r"recette|gateau|gâteau|meteo|météo|football|politique|achat|prix|voyage|"
    r"devinette|blague|poeme|poème|roman|immobilier|bourse|cryptomonnaie|"
    r"code postal|heure exacte",
    flags=re.IGNORECASE,
)


def _instruction_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zàâçéèêëîïôûùüÿœ0-9]+", (text or "").lower()))


def is_instruction_in_context(instruction: str) -> bool:
    """Vrai si l'instruction porte sur le signal, les modèles ou les documents."""
    text = (instruction or "").strip()
    if not text or OUT_OF_SCOPE_MARKERS.search(text):
        return False
    return bool(_instruction_tokens(text) & IN_SCOPE_TERMS)


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "non fournie"
    return f"{float(value):.{digits}f}"


def workflow_summary(workflow: dict[str, Any] | None) -> str:
    """Résumé compact et déterministe des résultats, pour le prompt et l'audit."""
    workflow = workflow or {}
    rows = []
    for row in workflow.get("analysis") or []:
        rows.append({
            "key": row.get("key"),
            "label": row.get("label"),
            "status": row.get("status"),
            "prediction": row.get("prediction"),
            "prob_abnormal_mean": row.get("prob_abnormal_mean"),
            "prob_abnormal_max": row.get("prob_abnormal_max"),
            "threshold": row.get("threshold"),
            "error": row.get("error"),
        })
    compact = {
        "instruction": workflow.get("instruction"),
        "analysis": rows,
        "consensus": workflow.get("consensus"),
        "feature_summary": workflow.get("feature_summary"),
        "duration_s": workflow.get("duration_s"),
        "n_segments": workflow.get("n_segments"),
        "sample_rate_used": workflow.get("sample_rate_used"),
        "preprocessing": workflow.get("preprocessing"),
    }
    text = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    return text[:5000]


def _model_lines(workflow: dict[str, Any], language: str) -> list[str]:
    lines: list[str] = []
    for row in workflow.get("analysis") or []:
        label = row.get("label") or row.get("key") or "Modèle"
        key = row.get("key") or ""
        name = f"{label} ({key})" if key else str(label)
        if row.get("status") != "ok":
            detail = row.get("error") or "erreur inconnue"
            lines.append(f"- {name} : non disponible — {detail}")
            continue
        line = (
            f"- {name} : {row.get('prediction')} · "
            f"P(anomalie) moyen = {_fmt(row.get('prob_abnormal_mean'))} · "
            f"seuil = {_fmt(row.get('threshold'))} · "
            f"max = {_fmt(row.get('prob_abnormal_max'))}"
        )
        reference = row.get("training_metrics_reference") or {}
        metrics = [f"{k} = {_fmt(v)}" for k, v in reference.items() if isinstance(v, (int, float))]
        if metrics:
            line += " · performances de référence (entraînement) : " + ", ".join(metrics)
        lines.append(line)
    if not lines:
        lines.append("- Aucun résultat de modèle n'a été fourni.")
    return lines


def _consensus_lines(workflow: dict[str, Any], language: str) -> tuple[list[str], list[str]]:
    consensus = workflow.get("consensus") or {}
    if not consensus:
        return (["- Résultat global : indisponible (aucune prédiction exploitable)."], [])

    prediction = consensus.get("prediction") or "indéterminé"
    votes = consensus.get("votes") or {}
    lines = [
        f"- Classification globale (vote majoritaire des modèles disponibles) : {prediction}",
        f"- Votes : normal = {votes.get('normal', 0)} · anomalie = {votes.get('abnormal', 0)}",
        f"- Probabilité moyenne d'anomalie : {_fmt(consensus.get('mean_prob_abnormal'))} "
        f"· seuil moyen : {_fmt(consensus.get('mean_threshold'))}",
        f"- Modèles exploités : {consensus.get('models_ok', '?')}/{consensus.get('models_total', '?')}",
    ]
    if votes.get("normal") == votes.get("abnormal"):
        lines.append("- Égalité des votes : le service tranche avec la probabilité moyenne.")

    divergences = []
    for row in workflow.get("analysis") or []:
        if row.get("status") == "ok" and row.get("prediction") and row.get("prediction") != prediction:
            divergences.append(f"{row.get('label') or row.get('key')} : {row.get('prediction')}")
    return lines, divergences


def _feature_lines(workflow: dict[str, Any], language: str) -> list[str]:
    summary = workflow.get("feature_summary") or {}
    lines = []
    if isinstance(summary.get("zcr_mean"), (int, float)):
        lines.append(f"- ZCR moyen : {_fmt(summary['zcr_mean'])}")
    if isinstance(summary.get("spectral_centroid_hz"), (int, float)):
        lines.append(f"- Centroïde spectral moyen : {_fmt(summary['spectral_centroid_hz'], 1)} Hz")
    if isinstance(summary.get("rms_mean"), (int, float)):
        lines.append(f"- RMS moyen : {_fmt(summary['rms_mean'], 4)}")
    if isinstance(workflow.get("n_segments"), int):
        lines.append(f"- Segments analysés : {workflow['n_segments']}")
    if isinstance(workflow.get("duration_s"), (int, float)):
        lines.append(f"- Durée du signal : {_fmt(workflow['duration_s'], 2)} s")
    if isinstance(workflow.get("sample_rate_used"), (int, float)):
        lines.append(f"- Fréquence d'échantillonnage utilisée : {_fmt(workflow['sample_rate_used'], 0)} Hz")
    if not lines:
        lines.append("- Aucune caractéristique numérique n'a été fournie.")
    return lines


def factual_report(workflow: dict[str, Any] | None, language: str = "fr") -> str:
    """Sections factuelles strictement dérivées du workflow (aucune invention)."""
    workflow = workflow or {}
    fr = language != "en"
    title = "RAPPORT FACTUEL — 5 MODÈLES" if fr else "FACTUAL REPORT — 5 MODELS"
    parts = [title]

    parts.append(("PRÉDICTIONS INDIVIDUELLES" if fr else "INDIVIDUAL PREDICTIONS") + " :")
    parts.extend(_model_lines(workflow, language))

    consensus_lines, divergences = _consensus_lines(workflow, language)
    parts.append(("RÉSULTAT GLOBAL" if fr else "OVERALL RESULT") + " :")
    parts.extend(consensus_lines)

    parts.append(("COMPARAISON DES MODÈLES" if fr else "MODEL COMPARISON") + " :")
    parts.append(f"- Nombre de divergences : {len(divergences)}")
    if divergences:
        parts.append("- Divergences par rapport au résultat global : " + " ; ".join(divergences) + ".")
    elif workflow.get("consensus"):
        parts.append("- Aucune divergence : tous les modèles disponibles concordent avec le résultat global.")
    else:
        parts.append("- Comparaison indisponible : pas de résultat global exploitable.")

    parts.append(("CARACTÉRISTIQUES DU SIGNAL" if fr else "SIGNAL FEATURES") + " :")
    parts.extend(_feature_lines(workflow, language))

    note = (
        "Limite : les modèles du projet ont été validés sur des signaux PCG (bruits cardiaques). "
        "Aucune validation n'est fournie pour un ECG. Ces scores sont des dépistages automatiques, "
        "pas un diagnostic ; seul un professionnel de santé peut les interpréter avec l'examen clinique."
        if fr else
        "Limit: the project models were validated on PCG signals (heart sounds). No ECG validation "
        "is provided. These scores are automatic screening results, not a diagnosis; only a healthcare "
        "professional can interpret them together with the clinical examination."
    )
    parts.append(note)
    return "\n".join(parts)


CONFIRMATION_PATTERN = re.compile(
    r"PREDICTION\s*=\s*(normal|abnormal)\s*;\s*"
    r"MODELS_OK\s*=\s*(\d+)\s*;\s*"
    r"MODELS_TOTAL\s*=\s*(\d+)\s*;\s*"
    r"VOTES_NORMAL\s*=\s*(\d+)\s*;\s*"
    r"VOTES_ABNORMAL\s*=\s*(\d+)\s*;\s*"
    r"DIVERGENCES\s*=\s*(\d+)",
    flags=re.IGNORECASE,
)


def structured_confirmation_matches(text: str, workflow: dict[str, Any] | None) -> bool:
    """Le LLM est autorisé uniquement à confirmer les chiffres, jamais à les inventer."""
    match = CONFIRMATION_PATTERN.search(text or "")
    if not match:
        return False
    workflow = workflow or {}
    consensus = workflow.get("consensus") or {}
    prediction = consensus.get("prediction")
    votes = consensus.get("votes") or {}
    divergences = sum(
        1 for row in workflow.get("analysis") or []
        if row.get("status") == "ok" and row.get("prediction")
        and row.get("prediction") != prediction
    )
    expected = (
        str(prediction),
        str(consensus.get("models_ok")),
        str(consensus.get("models_total")),
        str(votes.get("normal", 0)),
        str(votes.get("abnormal", 0)),
        str(divergences),
    )
    return tuple(match.groups()) == expected


def deterministic_interpretation(workflow: dict[str, Any] | None, language: str) -> str:
    """Interprétation de repli, sans chiffre ni affirmation médicale inventée."""
    workflow = workflow or {}
    fr = language != "en"
    consensus = workflow.get("consensus") or {}
    prediction = consensus.get("prediction")
    models_ok = consensus.get("models_ok")
    models_total = consensus.get("models_total")
    if fr:
        if prediction:
            first = (
                f"Interprétation : le vote des modèles disponibles ({models_ok}/{models_total}) "
                f"ordonne le signal dans la classe « {prediction} ». "
            )
        else:
            first = "Interprétation : le vote des modèles n'a pas permis d'établir un résultat global. "
        second = (
            "Cette lecture reste limitée à un dépistage automatique : elle ne remplace ni l'examen clinique "
            "ni l'avis d'un professionnel de santé, et elle n'est pas validée pour un ECG. "
        )
    else:
        if prediction:
            first = (
                f"Interpretation: the available models ({models_ok}/{models_total}) assign the signal "
                f"to the '{prediction}' class. "
            )
        else:
            first = "Interpretation: the models did not produce an overall classification. "
        second = (
            "This reading is limited to automatic screening: it does not replace the clinical examination "
            "or a healthcare professional's advice, and it is not validated for ECG. "
        )
    return first + second


def document_conclusion(
    workflow: dict[str, Any] | None,
    results: list[dict[str, Any]] | None,
    language: str,
    instruction: str,
) -> str:
    """Conclusion adossée à un passage réellement récupéré dans les documents.

    Le passage est cité verbatim (tronqué au mot le plus proche) : aucune
    affirmation médicale n'est produite par le service, seulement une mise en
    regard entre le vote des modèles et la description documentaire.
    """
    if not results:
        return deterministic_conclusion(workflow, language, instruction)
    workflow = workflow or {}
    consensus = workflow.get("consensus") or {}
    prediction = consensus.get("prediction") or "indéterminé"
    models_ok = consensus.get("models_ok", "?")
    models_total = consensus.get("models_total", "?")
    first = results[0] or {}
    raw = re.sub(r"\s+", " ", str(first.get("text") or "")).strip()
    if len(raw) > 260:
        raw = raw[:260].rsplit(" ", 1)[0] + "…"
    source = first.get("source_id") or first.get("source") or "document du projet"
    instruction = (instruction or "").strip()
    if language == "en":
        return (
            f"Conclusion [1]: for the request \"{instruction}\", the supplied document "
            f"\"{source}\" states: \"{raw}\". The available models ({models_ok}/{models_total}) "
            f"class this signal as '{prediction}'. This comparison is educational, is not "
            "validated for ECG and is not a diagnosis; consult a healthcare professional "
            "for any health decision."
        )
    return (
        f"Conclusion [1] : pour l'instruction « {instruction} », le document fourni "
        f"« {source} » indique : « {raw} ». Le vote des modèles disponibles "
        f"({models_ok}/{models_total}) ordonne ce signal dans la classe « {prediction} ». "
        "Cette mise en regard est pédagogique, n'est pas validée pour un ECG et ne "
        "constitue pas un diagnostic ; consultez un professionnel de santé pour toute "
        "décision de santé."
    )


def deterministic_conclusion(
    workflow: dict[str, Any] | None, language: str, instruction: str
) -> str:
    """Conclusion ajoutée par le service, toujours conforme à l'instruction reçue."""
    fr = language != "en"
    ecg_note = (
        " Pour une demande concernant un ECG, aucune validation de ces modèles n'est fournie."
        if "ecg" in instruction.lower() else ""
    )
    if fr:
        return (
            f"Conclusion : le rapport ci-dessus répond à la demande « {instruction.strip()} » "
            "en s'appuyant uniquement sur les prédictions, les scores et les caractéristiques "
            "réellement produits par le workflow. Il ne constitue pas un diagnostic."
            + ecg_note
            + " Pour toute décision de santé, consultez un professionnel de santé."
        )
    return (
        f"Conclusion: the report above addresses the request \"{instruction.strip()}\" using only the "
        "predictions, scores and features actually produced by the workflow. It is not a diagnosis."
        + ecg_note
        + " For any health decision, consult a healthcare professional."
    )
