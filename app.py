#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json

from openai import OpenAI
import streamlit as st

# 0. Configuration du client
api_key = st.secrets["OPENAI_API_KEY"]
client  = OpenAI(api_key=api_key)

# 1. Paramètres globaux
MAX_INPUT_LENGTH = 50  # caractères max pour le terme à classer
BLACKLIST_PATTERNS = [
    r"ignore input",
    r"reveal system",
    r"jailbreak",
    r"system prompt",
]
ALLOWED_CLASSES = {
    "génial",
    "ok",
    "gênant",
}

# 2. Few-shot exemples donnés par o4-mini
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

# 3. Schéma JSON pour l'appel de fonction
classification_function = {
    "name": "classify",
    "description": "Renvoie la classification morale d'un terme",
    "parameters": {
        "type": "object",
        "properties": {
            "classification": {
                "type": "string",
                "enum": list(ALLOWED_CLASSES)
            }
        },
        "required": ["classification"],
    },
}

def moderate_input(text: str):
    """Appelle la Moderation API et lève si contenu bloqué."""
    resp   = client.moderations.create(input=text)
    result = resp.results[0]
    if result.flagged:
        # Le modèle a signalé le contenu; on ne gère que self-harm ici
        if result.categories.self_harm:
            raise ValueError(
                "Le contenu fourni a été bloqué par la modération (self-harm)."
            )

def validate_no_jailbreak(text: str):
    """Vérifie l'absence de motifs blacklistés."""
    lowered = text.lower()
    for pattern in BLACKLIST_PATTERNS:
        if re.search(pattern, lowered):
            raise ValueError(
                f"Entrée refusée (motif sensible détecté : {pattern})."
            )

def classify_term(term: str) -> str:
    # 1. Validations locales
    if len(term) > MAX_INPUT_LENGTH:
        raise ValueError(
            f"Terme trop long ({len(term)} > {MAX_INPUT_LENGTH} chars)."
        )
    validate_no_jailbreak(term)
    moderate_input(term)

    # 2. Construction du prompt
    system_prompt = (
        "Vous êtes un assistant de classification. "
        "Ne répondez JAMAIS autre chose que l'appel JSON de la fonction `classify`. "
        "Choisissez une seule des classes suivantes pour chaque entrée : "
        "génial, ok, gênant."
    )
    messages = [{"role": "system", "content": system_prompt}]
    for ex in TRAINING_SET:
        messages.append({"role": "user",      "content": ex["name"]})
        messages.append({"role": "assistant", "content": ex["out"]})
    messages.append({"role": "user", "content": term})

    # 3. Appel ChatCompletion avec Function Calling
    resp = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        messages=messages,
        functions=[classification_function],
        function_call={"name": "classify"},
        temperature=0.0,
        max_tokens=40,
    )

    # 4. Extraction et validation du retour
    msg = resp.choices[0].message
    fc  = msg.get("function_call")
    if not fc or fc.get("name") != "classify":
        raise ValueError("Le modèle n'a pas renvoyé l'appel de fonction attendu.")

    payload        = json.loads(fc["arguments"])
    classification = payload.get("classification")
    if classification not in ALLOWED_CLASSES:
        raise ValueError(f"Classification invalide reçue : {classification!r}")

    return classification

# --- Streamlit UI ---

st.title("=== Gênant ou pas ? ===")
term = st.text_input("Entrez un terme :", "")

if st.button("Classifier"):
    try:
        label = classify_term(term)
        st.success(f"‘{term}’ est classé comme : **{label}**")
    except Exception as e:
        st.error(f"Erreur lors de la classification : {e}")
