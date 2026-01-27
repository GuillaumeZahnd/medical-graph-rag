import os
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase, exceptions


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def smoke_test() -> None:
    load_dotenv()
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    if not password:
        logging.error("NEO4J_PASSWORD not found.")
        return

    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:

            # Check basic connectivity
            driver.verify_connectivity()
            logging.info("✅ Connectivity: Success.")

            # Check for APOC plugin (for ingestion)
            with driver.session() as session:
                apoc_result = session.run("RETURN apoc.version() AS version")
                logging.info(f"✅ APOC Plugin: Active (v{apoc_result.single()['version']}).")

            # Check Graph Data Science (GDS) plugin (for GraphRAG analytics)
                gds_result = session.run("RETURN gds.version() AS version")
                logging.info(f"✅ GDS Plugin: Active (v{gds_result.single()['version']}).")



    except exceptions.ServiceUnavailable:
        logging.error("❌ Connectivity: Failed. Is Neo4j running?")

    except exceptions.AuthError:
        logging.error("❌ Auth: Failed. Check credentials.")

    except Exception as e:
        err_msg = str(e)
        if "apoc.version" in err_msg:
            logging.error("❌ APOC: Missing.")
        elif "gds.version" in err_msg:
            logging.error("❌ GDS: Missing.")
        else:
            logging.error(f"❌ Unexpected Error: {e}")


if __name__ == "__main__":
    smoke_test()
