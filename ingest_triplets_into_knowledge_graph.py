import os
from pathlib import Path
import pandas as pd
import logging
from typing import Any
from dotenv import load_dotenv
from neo4j import GraphDatabase


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD")
if not PASSWORD:
    raise ValueError("NEO4J_PASSWORD not found in environment variables.")

AUTH = (USER, PASSWORD)

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "logs" / "triplets.csv"


class KnowledgeGraphIngesto:
    def __init__(self, uri: str, auth: tuple[str, str], database: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.database = database

    def close(self) -> None:
        self.driver.close()


    def configure_database(self) -> None:
        """Initialize schema with constraints."""
        with self.driver.session(database=self.database) as session:
            session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            logging.info("Database constraints verified.")


    def ingest_batch(self, batch: list[dict[str, Any]]) -> None:
        """
        Ingest a batch of triplets.

        Args:
            batch: List of dictionaries of type [{'subject': 'Entity A', 'predicate': 'Predicate', 'object': 'Entity B'}, ...].
        """
        query = """
        UNWIND $rows AS row
        MERGE (s:Entity {name: row.subject})
        MERGE (o:Entity {name: row.object})
        WITH s, o, row
        CALL apoc.merge.relationship(s, row.predicate, {}, {}, o, {}) YIELD rel
        RETURN count(rel)
        """
        with self.driver.session(database=self.database) as session:
            session.execute_write(lambda tx: tx.run(query, rows=batch))


def main():
    ingestor = KnowledgeGraphIngesto(uri=URI, auth=AUTH, database="neo4j")
    ingestor.configure_database()

    dataframe = pd.read_csv(CSV_FILE)
    logging.info(f"Starting ingestion of {len(dataframe)} triplets...")

    batch_size = 500
    for i in range(0, len(dataframe), batch_size):
        batch = dataframe.iloc[i:i+batch_size].to_dict('records')
        ingestor.ingest_batch(batch)
        logging.info(f"Processed {min(i + batch_size, len(dataframe))} rows...")

    ingestor.close()


if __name__ == "__main__":
    main()
