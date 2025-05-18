#Contient les listes PLACEHOLDERS et TRAINING_SET utilisées par app.py.

# Placeholders proposés à l'utilisateur
PLACEHOLDERS = [
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

# Few‑shot exemples donnés par o4‑mini
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
    {"name": "L’intelligence artificielle", "out": "génial"},  # ça prêche pour sa paroisse
    {"name": "La lecture de romans", "out": "ok"},
    {"name": "Les chaînes de Ponzi", "out": "gênant"},
    {"name": "Les énergies renouvelables", "out": "génial"},
    {"name": "Le harcèlement scolaire", "out": "gênant"},
    {"name": "La cuisine végétarienne", "out": "ok"},
]