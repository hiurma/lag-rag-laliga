import os
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DB_URL = os.getenv("DB_URL", "sqlite:///data/laliga.sqlite")
