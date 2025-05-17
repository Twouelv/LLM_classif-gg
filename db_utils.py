import sqlite3
from datetime import datetime

# Base de données privée pour votes
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
    """Enregistre un vote dans la base locale, côté serveur."""
    tz = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO votes (term, vote, timestamp) VALUES (?, ?, ?)",
        (term, vote, tz)
    )
    conn.commit()