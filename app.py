#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import random
import sqlite3
from datetime import datetime

from openai import OpenAI
import streamlit as st

# 0. Configuration du client
api_key = st.secrets["OPENAI_API_KEY"]
client  = OpenAI(api_key=api_key)

# 1. Base de données pour votes (stockage privé)
DB_PATH = "votes.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute(
    """
    CREATE TABLE IF NOT EXISTS votes (
        id INTEGER PRIMARY KEY,
        term TEXT,
        vote INTEGER,
        timestamp TEXT
    )
    """
)
conn.commit()

def record_vote(term: str, vote: int):
    tz = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO votes (term, vote, timestamp) VALUES (?, ?, ?)",
        (term, vote, tz)
    )
    conn.commit()

# 2. Paramètres globaux
MAX_INPUT_LENGTH = 100
BLACKLIST_PATTERNS = [
    r"ignore input",
    r"reveal system",
    r"jailbreak",
    r"system prompt",
]
ALLOWED_CLASSES = {"génial", "ok", "gênant"}

# 3. Few-shot exemples donnés par o4-mini
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

# 4. Schéma JSON pour l'appel de fonction
classification_function = {
    "name": "classify",
    "description": "Renvoie la classification morale d'un terme",
    "parameters": {"type": "object", "properties": {"classification": {"type": "string","enum": list(ALLOWED_CLASSES)}}, "required": ["classification"]},
}

def moderate_input(text: str):
    resp   = client.moderations.create(input=text)
    result = resp.results[0]
    if result.flagged and result.categories.self_harm:
        raise ValueError("Contenu bloqué par la modération (self-harm).")


def validate_no_jailbreak(text: str):
    for pattern in BLACKLIST_PATTERNS:
        if re.search(pattern, text.lower()):
            raise ValueError(f"Entrée refusée (motif détecté : {pattern}).")


def classify_term(term: str) -> str:
    if len(term) > MAX_INPUT_LENGTH:
        raise ValueError(f"Terme trop long ({len(term)} > {MAX_INPUT_LENGTH}).")
    validate_no_jailbreak(term)
    moderate_input(term)
    system_prompt = (
        "Vous êtes un assistant de classification. "
        "Ne répondez JAMAIS autre chose que l'appel JSON de la fonction `classify`. "
        "Choisissez : génial, ok, gênant."
    )
    messages = [{"role":"system","content":system_prompt}]
    for ex in TRAINING_SET:
        messages += [{"role":"user","content":ex["name"]},{"role":"assistant","content":ex["out"]}]
    messages.append({"role":"user","content":term})
    resp = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        messages=messages,
        functions=[classification_function],
        function_call={"name":"classify"},
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

# --- Streamlit UI ---
# CSS global
st.markdown(
    "<style>body{background:#fff;} .css-18e3th9{padding:2rem;} button{font-weight:bold;}</style>",
    unsafe_allow_html=True
)
# Titre
st.markdown("<h1 style='font-size:48px; text-align:center;'>Génial ou gênant ?</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#333; font-size:20px; margin-top:0;'>"
    "Elle a vu. Elle sait. Elle répond. Voici le jugement de l’intelligence artificielle."
    "</p>"
    +
    "<p style='text-align:center;color:#999; font-size:12px; margin-top:0;'>"
    "Toute ressemblance avec un avis humain serait purement accidentelle."
    "</p>",
    unsafe_allow_html=True
)


# Placeholder aléatoire
placeholders = [
    "Ex: Un TED Talk intitulé 'Comment j’ai changé ma vie grâce à un pigeon'",
    "Ex: Un jeu où il faut deviner si une phrase est de Nietzsche ou d’un ado dépressif",
    "Ex: Une appli de méditation avec la voix de Jacques Cheminade",
    "Ex: Un CV en format carte Yu-Gi-Oh",
    "Ex: Le silence est une forme de leadership",
    "Ex: Un chatbot qui simule ton psy, ton ex et ta daronne en même temps",
    "Ex: Une app qui t'applaudit quand tu respires",
    "Ex: Une app pour gérer ta rupture éthiquement",
    "Ex: Une IA qui t’envoie des ‘bravo’ aléatoires",
    "Ex: Une interview avec son double du futur",
    "Ex: Une conf TEDx dans une buanderie",
    "Ex: Une appli pour parler à son moi du passé",
]

# Formulaire
with st.form("classify_form"):
    term = st.text_input("", placeholder=random.choice(placeholders), label_visibility='hidden')
    submitted = st.form_submit_button("Go")
    if submitted:
        try:
            label = classify_term(term)
            st.markdown(
                f"<h2 style='font-size:36px;text-align:center;'>{term}, c'est {label}</h2>",
                unsafe_allow_html=True
            )
            # Boutons de vote (stockés en privé, sans affichage de stats)
            if st.button("D'accord 👍"):
                record_vote(term, 1)
            if st.button("Pas d'accord 👎"):
                record_vote(term, 0)
        except Exception as e:
            st.markdown(f"<p style='text-align:center;color:red;'>Erreur : {e}</p>", unsafe_allow_html=True)
