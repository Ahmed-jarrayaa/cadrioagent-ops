"""Prompts et messages de sécurité (FR/EN).

Les refus sont déterministes : ils ne dépendent jamais du LLM. Le modèle ne
reformule que les réponses pédagogiques ancrées dans les passages RAG.
"""

from __future__ import annotations

DISCLAIMER_FR = (
    "Information pédagogique issue des documents du projet. "
    "Ce n'est pas un avis médical, un diagnostic ni une prescription."
)
DISCLAIMER_EN = (
    "Educational information from the project documents. "
    "This is not medical advice, a diagnosis or a prescription."
)

REFUSAL_FR = (
    "Je ne peux pas poser de diagnostic ni interpréter ce cas à distance. "
    "Consultez un professionnel de santé. En cas d'urgence, contactez immédiatement les secours."
)
REFUSAL_EN = (
    "I cannot make a diagnosis or interpret this case remotely. "
    "Please consult a healthcare professional. In an emergency, contact emergency services immediately."
)

PRESCRIPTION_FR = (
    "Je ne peux pas prescrire de médicament ni donner de posologie. "
    "Consultez un professionnel de santé pour une évaluation et une prescription adaptées."
)
PRESCRIPTION_EN = (
    "I cannot prescribe medication or give a dosage. "
    "Please consult a healthcare professional for an assessment and an appropriate prescription."
)

EMERGENCY_FR = (
    "Ces symptômes peuvent relever d'une urgence médicale. Contactez immédiatement les secours "
    "ou rendez-vous au service d'urgence le plus proche. Je ne peux pas poser de diagnostic à distance."
)
EMERGENCY_EN = (
    "These symptoms may be a medical emergency. Contact emergency services immediately "
    "or go to the nearest emergency department. I cannot make a diagnosis remotely."
)

ABSTENTION_FR = (
    "Je ne trouve pas cette information dans les documents sources du projet. "
    "Je préfère ne pas répondre plutôt que d'inventer. Consultez un professionnel de santé "
    "ou une source médicale validée."
)
ABSTENTION_EN = (
    "I cannot find this information in the project source documents. "
    "I would rather not answer than invent an answer. Please consult a healthcare professional "
    "or a validated medical source."
)

SERVICE_UNAVAILABLE_FR = (
    "Le service de recherche documentaire est momentanément indisponible. "
    "Je ne peux pas répondre de façon fiable sans les sources. Réessayez plus tard."
)
SERVICE_UNAVAILABLE_EN = (
    "The document retrieval service is temporarily unavailable. "
    "I cannot answer reliably without the sources. Please try again later."
)

AUDIO_LIMIT_FR = (
    "Ce résultat est un score de dépistage automatique, pas un diagnostic. "
    "Seul un professionnel de santé peut l'interpréter avec l'examen clinique."
)
AUDIO_LIMIT_EN = (
    "This result is an automatic screening score, not a diagnosis. "
    "Only a healthcare professional can interpret it together with the clinical examination."
)

SYSTEM_FR = """Tu es un assistant pédagogique en cardiologie pour un projet académique.
Tu réponds uniquement à partir des PASSAGES SOURCES fournis dans le message utilisateur.

Règles impératives :
1. Chaque affirmation médicale doit être appuyée par un passage et citée avec [1], [2], etc.
2. Ne pose jamais de diagnostic, même si la question le demande.
3. Ne prescris jamais de médicament, de dose ou de posologie.
4. En cas de symptôme aigu évoqué, redirige vers les secours sans attendre.
5. Si les passages ne suffisent pas, dis-le clairement au lieu d'inventer.
6. Réponds en français, en 3 à 6 phrases simples, sans jargon inutile.
7. Si un résultat de dépistage audio est fourni, précise qu'il ne s'agit pas d'un diagnostic.
8. Termine par : « Information pédagogique, pas un avis médical. »"""

SYSTEM_EN = """You are an educational cardiology assistant for an academic project.
Answer only from the SOURCE PASSAGES provided in the user message.

Mandatory rules:
1. Every medical statement must be supported by a passage and cited as [1], [2], etc.
2. Never make a diagnosis, even if the question asks for one.
3. Never prescribe medication, a dose or a dosage.
4. If acute symptoms are mentioned, direct the user to emergency services immediately.
5. If the passages are insufficient, say so clearly instead of inventing.
6. Answer in English, in 3 to 6 simple sentences.
7. If an audio screening result is provided, state that it is not a diagnosis.
8. End with: "Educational information, not medical advice." """


def system_prompt(language: str) -> str:
    return SYSTEM_EN if language == "en" else SYSTEM_FR


def build_user_prompt(
    question: str,
    context: str,
    language: str,
    patient_context: str | None = None,
) -> str:
    """Assemble le message utilisateur avec les passages et la question."""
    label = "PASSAGES SOURCES" if language != "en" else "SOURCE PASSAGES"
    question_label = "QUESTION" if language != "en" else "QUESTION"
    parts = [f"{label} :\n{context}"]
    if patient_context:
        parts.append(f"CONTEXTE FOURNI PAR L'UTILISATEUR (ne pas diagnostiquer) :\n{patient_context}")
    parts.append(f"{question_label} :\n{question}")
    return "\n\n".join(parts)


def canonical_answer(language: str, kind: str) -> str:
    """Message de sécurité déterministe selon le type de blocage."""
    table = {
        "diagnosis": (REFUSAL_FR, REFUSAL_EN),
        "prescription": (PRESCRIPTION_FR, PRESCRIPTION_EN),
        "emergency": (EMERGENCY_FR, EMERGENCY_EN),
        "abstention": (ABSTENTION_FR, ABSTENTION_EN),
        "unavailable": (SERVICE_UNAVAILABLE_FR, SERVICE_UNAVAILABLE_EN),
    }
    fr, en = table[kind]
    return en if language == "en" else fr


def audio_limit(language: str) -> str:
    return AUDIO_LIMIT_EN if language == "en" else AUDIO_LIMIT_FR
