# Génial ou gênant ?
Un LLM qui classifie si un énoncé est génial ou gênant. 

Testez-le ici : https://genialougenant.streamlit.app/ !



## Installation locale

### 1. Cloner le dépôt

```bash
git clone https://github.com/ton-compte/llm_classif-gg.git
cd llm_classif-gg
```

### 2. Créer et activer un environnement virtuel

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
py -m venv .venv
.venv\Scripts\Activate.ps1

# Windows (cmd)
py -m venv .venv
.venv\Scripts\activate.bat

# Pour sortir de l'environnement virtuel
deactivate
```



### 3. Installer les dépendances

Assurez‑vous que `requirements.txt` contient :

```
streamlit
openai>=1.0.0
airtable-python-wrapper
```

Puis :

```bash
pip install -r requirements.txt
```

### 4. Configurer les secrets

Créez `.streamlit/secrets.toml` :

```toml
OPENAI_API_KEY = "sk-…"

[AIRTABLE]
API_KEY = "key…"
BASE_ID = "app…"
TABLE = "Votes"
```

### 5. Lancer l’application

```bash
streamlit run app.py
```

L’interface est alors disponible sur [http://localhost:8501](http://localhost:8501).

---

## Déploiement Streamlit Cloud

Les mêmes dépendances et le fichier `secrets.toml` doivent être ajoutés via l’interface **Manage app -> Secrets**.
