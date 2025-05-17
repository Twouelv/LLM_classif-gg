#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import random
import sqlite3
from datetime import datetime
from openai import OpenAI
import streamlit as st

# --- Configuration client OpenAI ---
api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

# --- Base de données votes (privé) ---
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

# --- Paramètres globaux ---
MAX_INPUT_LENGTH = 100
BLACKLIST_PATTERNS = [r"ignore input", r"reveal system", r"jailbreak", r"system prompt"]
ALLOWED_CLASSES = {"génial", "ok", "gênant"}
# Few-shot exemples donnés par o4-mini
TRAINING_SET = [
    {"name": "La désinformation en ligne", "out": "gênant"},
    {"name": "La robotique chirurgicale", "out": "génial"},
    {"name": "La randonnée en montagne", "out": "ok"},
    {"name": "Le bruit urbain", "out": "gênant"},
    {"name": "Les documentaires scientifiques", "out": "ok"},
    {"name": "Les voyages interstellaires", "out": "génial"},
    {"name": "Le recyclage des déchets", "out": "ok"},
    {"name": "Les spams par e-mail", "out": "gênant"},
    {"name": "La protection de la vie privée", "out": "génial"},
    {"name": "Le télétravail", "out": "ok"},
    {"name": "Les arnaques à la loterie", "out": "gênant"},
    {"name": "La musique classique", "out": "génial"},
    {"name": "Le piratage de comptes", "out": "gênant"},
    {"name": "L’intelligence artificielle", "out": "génial"}, #ça prêche pour sa paroisse
    {"name": "La lecture de romans", "out": "ok"},
    {"name": "Les chaînes de Ponzi", "out": "gênant"},
    {"name": "Les énergies renouvelables", "out": "génial"},
    {"name": "Le harcèlement scolaire", "out": "gênant"},
    {"name": "La cuisine végétarienne", "out": "ok"},
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

def moderate_input(text: str):
    resp = client.moderations.create(input=text)
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
        "Choisissez une seule des classes suivantes : génial, ok, gênant."
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

# --- Streamlit UI ---
# CSS pour boutons larges
st.markdown(
    "<style>div.stButton > button { width: 100%; padding: 1rem; font-size:1rem; }</style>",
    unsafe_allow_html=True
)
# Titre
st.markdown(
    "<h1 style='font-size:48px; text-align:center;'>Génial ou gênant ?</h1>",
    unsafe_allow_html=True
)
# Oraculobot millénaire
st.markdown(
    "<p style='text-align:center;color:#555;font-size:20px;margin-top:0;'>" +
    "Elle a vu. Elle sait. Elle répond. Voici le jugement de l’intelligence artificielle." +
    "</p>" +
    "<p style='text-align:center;color:#999;font-size:12px;margin-top:0;'>" +
    "Toute ressemblance avec un avis humain serait purement accidentelle." +
    "</p>",
    unsafe_allow_html=True
)

# Placeholders
placeholders = [
    "Ex: Un TED Talk intitulé 'Comment j’ai changé ma vie grâce à un pigeon'",
    "Ex: Un jeu où il faut deviner si une phrase est de Nietzsche ou d’un ado dépressif",
    "Ex: Une appli de méditation avec la voix de Jacques Cheminade",
    "Ex: Un CV en format carte Yu-Gi-Oh",
    "Ex: Le silence est une forme de leadership",
    "Ex: Un chatbot qui simule ton psy, ton ex et ta daronne",
    "Ex: Une app qui t'applaudit quand tu respires",
    "Ex: Une app pour gérer ta rupture éthiquement",
    "Ex: Une IA qui t’envoie des ‘bravo’ aléatoires",
    "Ex: Une interview avec son double du futur",
    "Ex: Une conf TEDx dans une buanderie",
    "Ex: Une appli pour parler à son moi du passé",
]

# Session state
if 'show_result' not in st.session_state:
    st.session_state.show_result = False
if 'voted' not in st.session_state:
    st.session_state.voted = False

# Callback à l'envoi de la saisie
def on_submit():
     st.session_state.term = st.session_state.user_input
     try:
         st.session_state.label = classify_term(st.session_state.user_input)
         st.session_state.show_result = True
         st.session_state.voted = False
     except Exception as e:
         st.error(f"Erreur : {e}")

# Champ de saisie avec callback (Enter ou click)
st.user_input = st.text_input(
    "Entrez votre terme :", 
    placeholder=random.choice(placeholders),
    key="user_input",
    on_change=on_submit
)

# Bouton manuel si préférence
if st.button("Go"):
    on_submit()

# Affichage résultat
if st.session_state.show_result:
    t = st.session_state.term
    l = st.session_state.label
    st.markdown(
        f"<h2 style='font-size:36px;text-align:center;'>{t}, c'est {l}</h2>",
        unsafe_allow_html=True
    )
    # Boutons côte à côte
    c1, c2 = st.columns(2, gap='large')
    if not st.session_state.get('voted', False):
        if c1.button("D'accord 👍"):
            record_vote(t, 1)
            st.session_state.voted = True
        if c2.button("Pas d'accord 👎"):
            record_vote(t, 0)
            st.session_state.voted = True
