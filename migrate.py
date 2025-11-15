# ops/migrate.py
import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import yaml

load_dotenv()
DB_PATH = os.getenv('DB_PATH', 'data/quant_terminal.db')
engine = create_engine(f'sqlite:///{DB_PATH}')
Session = sessionmaker(bind=engine)

def get_schema_version(session) -> int:
    result = session.execute(text("SELECT COUNT(*) FROM model_metadata WHERE model_name = 'schema'")).scalar()
    return result or 0

def init_schema():
    session = Session()
    version = get_schema_version(session)
    if version >= 1:
        print("Schema v1 exists. Skipping.")
        return

    # Run schema SQL (paste schema here or from file)
    schema_sql = """
    -- [Full SQL from above]
    """
    with engine.connect() as conn:
        conn.execute(text(schema_sql))
        conn.commit()

    # Log version
    session.execute(text("INSERT INTO model_metadata (model_name, version, params) VALUES ('schema', '1.0', '{}')"))
    session.commit()
    print("Schema v1 initialized.")

if __name__ == '__main__':
    if 'init' in sys.argv:
        init_schema()
