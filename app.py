import random
import streamlit as st
from openai import OpenAI
from classify_utils import classify_term
from db_utils import record_vote

# Initialisation du client
api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

# Styles CSS pour boutons
st.markdown(
    "<style>div.stButton > button { width: 100%; padding: 1rem; font-size:1rem; }</style>",
    unsafe_allow_html=True
)

# Titre et accroche
st.markdown(
    "<h1 style='font-size:48px; text-align:center;'>Génial ou gênant ?</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;color:#555;font-size:20px;margin-top:0;'>"
    "Elle a vu. Elle sait. Elle répond. Voici le jugement de l’intelligence artificielle."
    "</p>"
    "<p style='text-align:center;color:#999;font-size:12px;margin-top:0;'>"
    "Toute ressemblance avec un avis humain serait purement accidentelle."
    "</p>",
    unsafe_allow_html=True
)

# Placeholders aléatoires
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

# State
if 'show_result' not in st.session_state:
    st.session_state.show_result = False

# Saisie utilisateur
user_input = st.text_input(
    "Entrez votre terme :", 
    placeholder=random.choice(placeholders),
    key="user_input"
)
if st.button("Go"):
    try:
        label = classify_term(client, user_input)
        st.session_state.term = user_input
        st.session_state.label = label
        st.session_state.show_result = True
    except Exception as e:
        st.error(f"Erreur : {e}")

# Affichage du résultat et votes
if st.session_state.show_result:
    t = st.session_state.term
    l = st.session_state.label
    st.markdown(f"<h2 style='font-size:36px;text-align:center;'>{t}, c'est {l}</h2>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap='large')
    if c1.button("D'accord 👍"):
        record_vote(t, 1)
        st.session_state.show_result = False
    if c2.button("Pas d'accord 👎"):
        record_vote(t, 0)
        st.session_state.show_result = False
