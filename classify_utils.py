import re
import json
from openai import OpenAI
import streamlit as st
from data_sets import TRAINING_SET

# ------------------------------------------------------------------
# Constantes
# ------------------------------------------------------------------
MAX_INPUT_LENGTH   = 100
BLACKLIST_PATTERNS = [r"ignore input", r"reveal system", r"jailbreak", r"system prompt"]
ALLOWED_CLASSES    = {"génial", "ok", "gênant"}

classification_function = {
    "name": "classify",
    "description": "Renvoie la classification morale d'un terme",
    "parameters": {
        "type": "object",
        "properties": {
            "classification": {"type": "string", "enum": list(ALLOWED_CLASSES)}
        },
        "required": ["classification"],
    },
}

# Client OpenAI unique (chargé une seule fois)
_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ------------------------------------------------------------------
# Helpers internes
# ------------------------------------------------------------------

def _moderate_input(text: str) -> None:
    resp = _client.moderations.create(input=text)
    if resp.results[0].flagged and resp.results[0].categories.self_harm:
        raise ValueError("Contenu bloqué par la modération (self-harm).")


def _validate_no_jailbreak(text: str) -> None:
    lowered = text.lower()
    for pattern in BLACKLIST_PATTERNS:
        if re.search(pattern, lowered):
            raise ValueError("Entrée refusée (tentative jailbreak)")

# ------------------------------------------------------------------
# Fonction publique
# ------------------------------------------------------------------

def classify_term(term: str) -> str:
    """Retourne "génial", "ok" ou "gênant" pour le *term*."""
    if len(term) > MAX_INPUT_LENGTH:
        raise ValueError("Terme trop long")

    _validate_no_jailbreak(term)
    _moderate_input(term)

    system_prompt = (
        "Vous êtes un assistant de classification. "
        "Ne répondez que par l'appel JSON de la fonction `classify`. "
        "Choisissez : génial, ok, gênant."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for ex in TRAINING_SET:
        messages += [
            {"role": "user", "content": ex["name"]},
            {"role": "assistant", "content": ex["out"]},
        ]
    messages.append({"role": "user", "content": term})

    resp = _client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        messages=messages,
        functions=[classification_function],
        function_call={"name": "classify"},
        temperature=0.0,
        max_tokens=40,
    )

    fc = resp.choices[0].message.function_call
    if not fc or fc.name != "classify":
        raise ValueError("Réponse LLM invalide (function_call manquant)")

    classification = json.loads(fc.arguments)["classification"]
    if classification not in ALLOWED_CLASSES:
        raise ValueError("Classification hors liste")

    return classification