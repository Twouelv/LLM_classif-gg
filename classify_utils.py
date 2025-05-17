import re
import json
from openai import OpenAI

# Paramètres globaux
MAX_INPUT_LENGTH = 100  # caractères max pour le terme à classer
BLACKLIST_PATTERNS = [
    r"ignore input",
    r"reveal system",
    r"jailbreak",
    r"system prompt",
]
ALLOWED_CLASSES = {"génial", "ok", "gênant"}

# Few-shot exemples donnés par o4-mini
TRAINING_SET = [
    {"name": "La désinformation en ligne",      "out": "gênant"},
    {"name": "La robotique chirurgicale",       "out": "génial"},
    {"name": "La randonnée en montagne",        "out": "ok"},
    {"name": "Le bruit urbain",                 "out": "gênant"},
    {"name": "Les documentaires scientifiques", "out": "ok"},
    {"name": "Les voyages interstellaires",     "out": "génial"},
    {"name": "Le recyclage des déchets",        "out": "ok"},
    {"name": "Les spams par e-mail",            "out": "gênant"},
    {"name": "La protection de la vie privée",  "out": "génial"},
    {"name": "Le télétravail",                  "out": "ok"},
    {"name": "Les arnaques à la loterie",       "out": "gênant"},
    {"name": "La musique classique",            "out": "génial"},
    {"name": "Le piratage de comptes",          "out": "gênant"},
    {"name": "L’intelligence artificielle",     "out": "génial"},  # ça prêche pour sa paroisse
    {"name": "La lecture de romans",            "out": "ok"},
    {"name": "Les chaînes de Ponzi",            "out": "gênant"},
    {"name": "Les énergies renouvelables",      "out": "génial"},
    {"name": "Le harcèlement scolaire",         "out": "gênant"},
    {"name": "La cuisine végétarienne",         "out": "ok"},
]

# Schéma JSON pour function calling
classification_function = {
    "name": "classify",
    "description": "Renvoie la classification morale d'un terme",
    "parameters": {
        "type": "object",
        "properties": {"classification": {"type": "string", "enum": list(ALLOWED_CLASSES)}},
        "required": ["classification"],
    },
}

def moderate_input(client: OpenAI, text: str):
    """Appelle la Moderation API et lève si contenu bloqué."""
    resp = client.moderations.create(input=text)
    result = resp.results[0]
    if result.flagged and result.categories.self_harm:
        raise ValueError("Contenu bloqué par la modération (self-harm).")


def validate_no_jailbreak(text: str):
    """Vérifie l'absence de motifs blacklistés."""
    lowered = text.lower()
    for pattern in BLACKLIST_PATTERNS:
        if re.search(pattern, lowered):
            raise ValueError(f"Entrée refusée (motif détecté : {pattern}).")


def classify_term(client: OpenAI, term: str) -> str:
    """Classification d'un terme via ChatCompletion + Function Calling."""
    if len(term) > MAX_INPUT_LENGTH:
        raise ValueError(f"Terme trop long ({len(term)} > {MAX_INPUT_LENGTH}).")
    validate_no_jailbreak(term)
    moderate_input(client, term)

    system_prompt = (
        "Vous êtes un assistant de classification. "
        "Ne répondez JAMAIS autre chose que l'appel JSON de la fonction `classify`. "
        "Choisissez : génial, ok, gênant."
    )
    messages = [{"role": "system", "content": system_prompt}]
    for ex in TRAINING_SET:
        messages.append({"role": "user", "content": ex["name"]})
        messages.append({"role": "assistant", "content": ex["out"]})
    messages.append({"role": "user", "content": term})

    resp = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        messages=messages,
        functions=[classification_function],
        function_call={"name": "classify"},
        temperature=0.0,
        max_tokens=40,
    )
    fc = resp.choices[0].message.function_call
    if not fc or fc.name != "classify":
        raise ValueError("Appel de fonction classify attendu.")
    payload = json.loads(fc.arguments)
    cls = payload.get("classification")
    if cls not in ALLOWED_CLASSES:
        raise ValueError(f"Classification invalide reçue : {cls!r}")
    return cls