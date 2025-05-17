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
    {"name": "L’intelligence artificielle",     "out": "génial"}, #ça prêche pour sa paroisse
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
            "classification": {"type": "string","enum": list(ALLOWED_CLASSES)}
        },
        "required": ["classification"],
    },
}

def moderate_input(text: str):
    """Appelle la Moderation API et lève si contenu bloqué."""
    resp   = client.moderations.create(input=text)
    result = resp.results[0]
    if result.flagged and result.categories.self_harm:
        raise ValueError("Le contenu fourni a été bloqué par la modération (self-harm).")


def validate_no_jailbreak(text: str):
    """Vérifie l'absence de motifs blacklistés."""
    lowered = text.lower()
    for pattern in BLACKLIST_PATTERNS:
        if re.search(pattern, lowered):
            raise ValueError(f"Entrée refusée (motif sensible détecté : {pattern}).")


def classify_term(term: str) -> str:
    # Validations locales
    if len(term) > MAX_INPUT_LENGTH:
        raise ValueError(f"Terme trop long ({len(term)} > {MAX_INPUT_LENGTH}).")
    validate_no_jailbreak(term)
    moderate_input(term)

    # Construction du prompt
    system_prompt = (
        "Vous êtes un assistant de classification. "
        "Ne répondez JAMAIS autre chose que l'appel JSON de la fonction `classify`. "
        "Choisissez une seule des classes suivantes : génial, ok, gênant."
    )
    messages = [{"role": "system", "content": system_prompt}]
    for ex in TRAINING_SET:
        messages += [
            {"role": "user",      "content": ex["name"]},
            {"role": "assistant", "content": ex["out"]}
        ]
    messages.append({"role": "user", "content": term})

    # Appel ChatCompletion
    resp = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        messages=messages,
        functions=[classification_function],
        function_call={"name": "classify"},
        temperature=0.0,
        max_tokens=40,
    )

    # Extraction du résultat
    fc = resp.choices[0].message.function_call
    if fc is None or fc.name != "classify":
        raise ValueError("Attendu un appel de fonction classify.")

    payload = json.loads(fc.arguments)
    classification = payload.get("classification")
    if classification not in ALLOWED_CLASSES:
        raise ValueError(f"Classification invalide reçue : {classification!r}")

    return classification

# --- Streamlit UI avec formulaire ---

# Style global pour titre et résultats
st.markdown("<style>\nbody {background-color: #fff;}\n.css-18e3th9 {padding: 2rem;}\n</style>", unsafe_allow_html=True)
# Titre principal agrandi
st.markdown("<h1 style='font-size:48px; text-align:center; margin-bottom:0.2em;'>Gênant ou génial ?</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color: #666; margin-top:0;'>L'intelligence artificielle a la réponse</p>", unsafe_allow_html=True)

with st.form("classify_form"):
    term = st.text_input("Entrez un terme :", "", placeholder="Ex: Les bananes")
    submitted = st.form_submit_button("Go")
    if submitted:
        try:
            label = classify_term(term)
            # Affichage du résultat en gros texte
            st.markdown(
                f"<h2 style='font-size:36px; text-align:center;'>{term}, c'est {label}</h2>",
                unsafe_allow_html=True
            )
        except Exception as e:
            st.markdown(f"<p style='text-align:center; color:red;'>Erreur : {e}</p>", unsafe_allow_html=True)
