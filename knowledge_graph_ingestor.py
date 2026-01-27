import os
from pathlib import Path
import pandas as pd
import json
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase
from typing import Any

from llm_client import ChatClient
from schema import MedicalPredicate, TripletExtraction


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class KnowledgeGraphIngestor:
    """Engine that handles the transition from text chunks to graph database."""
    def __init__(self, uri: str, auth: tuple[str, str], database: str, llm_client: ChatClient, llm_model: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=auth, database=database)
        self.database = database
        self._llm_client = llm_client
        self._llm_model = llm_model


    def close(self) -> None:
        self.driver.close()


    def configure_database(self) -> None:
        """Initialize schema with constraints."""

        constraints = [
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT doc_name IF NOT EXISTS FOR (d:Document) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE"
        ]
        with self.driver.session(database=self.database) as session:
            for query in constraints:
                session.run(query)
            logging.info("Database constraints verified.")


    def ingest_chunks(self, chunks: list[dict[str, Any]], batch_size: int = 500) -> None:
        """
        Ingest a batch of text chunks and link them to their parent documents.

        Args:
            chunks: List of dictionaries representing processed document segments.
        """

        if not chunks:
            logging.warning("Warning: No chunks to ingest.")
            return

        query = """
        UNWIND $rows AS row
        MERGE (d:Document {name: row.source})
        MERGE (c:Chunk {id: row.id})
        ON CREATE SET
            c.text = row.data,
            c.hierarchy = row.hierarchy,
            c.created_at = timestamp()
        ON MATCH SET
            c.last_seen = timestamp()
        MERGE (c)-[:PART_OF]->(d)
        """
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                session.execute_write(lambda tx: tx.run(query, rows=batch))

        logging.info(f"Successfully ingested {len(chunks)} chunks.")


    def ingest_aliases(self, alias_lookup_map: dict[str, str]) -> None:
        """
        Ingest a mapping of synonyms to canonical entities.

        Args:
            alias_lookup_map: Dictionary that maps every messy name variation, alias, or synonym found in the text to a
            unique canonical name, for instance: {"hta": "Hypertension", "high bp": "Hypertension", "High Blood Pressure", "Hypertension", ...}
        """
        if not alias_lookup_map:
            logging.warning("Warning: No aliases to ingest.")
            return

        rows = [
            {"alias": str(k).strip(), "canonical": str(v).strip()}
            for k, v in alias_lookup_map.items()
        ]

        query = """
        UNWIND $rows AS row

        // Handle Canonical Node
        MERGE (c:Entity {name: row.canonical})
        ON CREATE SET c.is_canonical = true
        ON MATCH SET c.is_canonical = true

        // Handle Alias Node
        MERGE (a:Entity {name: row.alias})

        // Thread-safe relationship creation
        WITH a, c
        WHERE id(a) <> id(c)
        MERGE (a)-[:SYNONYM_OF]->(c)
        """

        with self.driver.session(database=self.database) as session:
            session.execute_write(lambda tx: tx.run(query, rows=rows))

        logging.info(f"Successfully ingested {len(rows)} alias mappings.")


    def ingest_triplets(self, triplets: list[dict[str, Any]], batch_size: int = 500) -> None:
        """
        Ingest a batch of semantic triplets using APOC for dynamic relationship types.

        Args:
            triplets: List of dictionaries of type [{'subject': 'Entity A', 'predicate': 'Predicate', 'object': 'Entity B'}, ...].
        """

        if not triplets:
            logging.warning("Warning: No triplets to ingest.")
            return

        processed_rows = []
        for t in triplets:
            processed_rows.append({
                "subject": str(t["subject"]).strip(),
                "object": str(t["object"]).strip(),
                # Normalize predicate: "Treated by" -> "TREATED_BY"
                "predicate": str(t["predicate"]).strip().replace(" ", "_").upper(),
                "source": t.get("source", "unknown")
            })

        query = """
        UNWIND $rows AS row
        MERGE (s:Entity {name: row.subject})
        MERGE (o:Entity {name: row.object})
        WITH s, o, row
        CALL apoc.merge.relationship(
            s,
            row.predicate,
            {},
            {source: row.source, last_updated: timestamp()},
            o,
            {}
        ) YIELD rel
        RETURN count(*) as total
        """
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(processed_rows), batch_size):
                batch = processed_rows[i : i + batch_size]
                session.execute_write(lambda tx: tx.run(query, rows=batch))

        logging.info(f"Successfully ingested {len(triplets)} triplets.")


    def _llm_call(self, system_prompt: str, user_content: str):
        """
        Helper to handle API calls with JSON formatting.
        """
        try:
            response = self._llm_client.chat.complete(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error during LLM call: {e}")
            return None


    def extract_entities(self, batch_text: list[str]) -> list[list[dict]]:
        """
        Perform batch Named Entity Recognition (NER) on text segments to identify medical concepts.
        Preliminary pass to extract entities, aiming to achieve a high recall.

        Args:
            batch_text: List of independent text segments/chunks to be processed.

        Returns:
            Nested list where each outer element corresponds to an input segment by index. Each inner list contains
            dictionaries representing the extracted entities, e.g., [{"name": "Pylera", "type": "Drug"}].
        """

        system_prompt = """
        You are an expert medical annotator.
        Your task is to extract all clinical entities from the provided text to build a medical knowledge graph.

        EXTRACT THE FOLLOWING CATEGORIES:
        1. DRUGS & TREATMENTS: Brand names, generic names, and therapeutic protocols (e.g., 'Pylera', 'Amoxicilline', 'Quadruple thérapie').
        2. DISEASES & CONDITIONS: Pathogens, syndromes, and specific diagnoses (e.g., 'Infection à H. pylori', 'Cancer gastrique').
        3. SYMPTOMS & SIGNS: Clinical manifestations (e.g., 'Dysgueusie', 'Carence en fer').
        4. TESTS & PROCEDURES: Diagnostic or monitoring actions (e.g., 'Endoscopie', 'Test respiratoire à l'urée').
        5. PATIENT POPULATIONS: Specific demographics or groups (e.g., 'Enfants <12 ans', 'Femmes enceintes', 'Seniors').
        6. CLINICAL STATUS/CONTEXT: Specific patient history or required conditions (e.g., 'Antécédent d'ulcère', 'Avant bypass gastrique', 'Traitement au long cours par IPP').

        CRITICAL RULES:
        - Extract entities exactly as they appear in the text, but prioritize the full clinical phrase (e.g., 'Carence en vitamine B12' instead of just 'Carence').
        - Do not extract vague terms like 'patients', 'doctor', or 'hospital' unless they are part of a specific population group.

        OUTPUT FORMAT:
        Return a JSON object with a 'results' key. Each entry must have a 'segment_id' matching the input.
        {
            "results": [
                {"segment_id": 0, "entities": [{"name": "Pylera", "type": "Drug"}]},
                ...
            ]
        }
        """
        user_content = "\n\n".join([f"SEGMENT {i}:\n{text}" for i, text in enumerate(batch_text)])

        response = self._llm_call(system_prompt=system_prompt, user_content=user_content)

        # Prevent crashes if "results" is missing
        if not response or "results" not in response:
            print("WARNING: Entity extraction returned no results.")
            return [[] for _ in range(len(batch_text))]

        results_map = {item['segment_id']: item['entities'] for item in response.get('results', [])}
        results = [results_map.get(i, []) for i in range(len(batch_text))]

        return results


    def resolve_entities(self, entities: list[dict]) -> list[dict]:
        """
        Perform Entity Resolution (ER) to map medical variations to canonical forms.
        Avoid synonyms and acronyms, semantically regroup entities (e.g., "Hypertension", "HTN", "High Blood Pressure").

        Args:
            entities: List of extracted entity dictionaries to normalize.

        Returns:
            List of normalized entity dictionaries of the same length and order as the input.
        """

        system_prompt = """
        You are a medical data architect specializing in Entity Resolution.
        For every entity provided in the input list, you must provide its canonical version.

        RULES:
        1. Do not consolidate the list. If I send 10 entities, you must return exactly 10 entities in the same order.
        2. For synonyms (e.g., 'HTA'), provide the standard French term (e.g., 'Hypertension').
        3. Keep the clinical category (Disease, Drug, etc.) accurate.

        OUTPUT FORMAT:
        Return a JSON object with a key 'normalized_entities' containing the list.
        Example:
        Input: [{"name": "HTA"}, {"name": "Stroke"}]
        Output: {"normalized_entities": [{"name": "Hypertension", "type": "Disease"}, {"name": "Accident Vasculaire Cérébral", "type": "Disease"}]}
        """
        user_content = f"Entities to normalize: {json.dumps(entities)}"

        response = self._llm_call(system_prompt, user_content)
        results = response.get('normalized_entities', [])

        # Verification whether the LLM returned the same number of entities that we gave as input.
        # This step is crucial to prevent the routine "zip(stric=True)" from the registry from crashing.
        if len(results) != len(entities):
            print(f"DEBUG: LLM returned {len(results)} but we expected {len(entities)}.")

            # If the LLM returned too few entities, we pad the results with "Unknown".
            while len(results) < len(entities):
                results.append({"name": "Unknown", "type": "Unknown"})

            # If the LLM returned too many (rare), we truncate.
            results = results[:len(entities)]

        return results


    def extract_triplets(self, source_text: str, entities: list[dict]) -> list[dict]:
        """
        Extract directed semantic relationships between validated medical entities.
        For example: ["Lisinopril", "FOLLOW_UP", "Kidney function test in 2 weeks"].

        Args:
            source_text: Raw text segment containing potential relationships.
            entities: List of normalized entities (canonical names) available for relationship mapping.

        Returns:
            List of validated triplets in the format: {"subject": str, "predicate": str, "object": str}.
        """

        # Create a mapping to recover keys from hallucinated descriptions
        value_to_key = {p.value: p.name for p in MedicalPredicate}

        # Prepare schema and constraints for the prompt
        predicate_info = "\n".join([f"- {p.name}: {p.value}" for p in MedicalPredicate])

        # Get the JSON schema to show Mistral exactly what we expect
        json_schema = json.dumps(TripletExtraction.model_json_schema(), indent=2)

        system_prompt = f"""
        You are a medical knowledge graph builder. Extract relationships between entities as triplets.

        STRICT DIRECTIONALITY RULES:
        1. INDICATED_FOR: [Drug/Treatment] -> [Disease]
        2. MANIFESTS_AS: [Disease] -> [Symptom/Sign]
        3. CONTRAINDICATED_WITH: [Drug] -> [Condition/Other Drug]
        4. REQUIRES_DIAGNOSTIC_TEST: [Disease] -> [Test]
        5. REQUIRES_MONITORING_TEST: [Drug] -> [Test]
        6. TREATMENT_OPTION: [Disease] -> [Drug/Protocol/Procedure]
        7. FOLLOW_UP_PLAN: [Treatment/Disease] -> [Schedule/Action]
        8. DEMOGRAPHICS: [Disease/Treatment] -> [Patient Population/Age Group]

        ALLOWED PREDICATES:
        {predicate_info}

        OUTPUT FORMAT:
        You must return a JSON object that adheres to this schema:
        {json_schema}
        """

        entity_names = [e['name'] for e in entities if 'name' in e]

        user_content = f"""
        TEXT: "{source_text}"

        VALID ENTITIES: {', '.join(entity_names)}

        Instructions:
        1. Only use the provided VALID ENTITIES for 'subject_entity' and 'object_entity'.
        2. Only use the ALLOWED PREDICATES for 'predicate'.
        3. If no relationships are found, return {{"triplets": []}}.
        """

        raw_json = self._llm_call(system_prompt, user_content)
        if not raw_json:
            return []

        # Pre-processing: convert descriptions back to Enum keys to satisfy Pydantic
        for triplet in raw_json["triplets"]:
            pred = triplet.get("predicate")
            if pred in value_to_key:
                triplet["predicate"] = value_to_key[pred]

        # Validate and parse the LLM output with Pydantic
        try:
            validated_data = TripletExtraction.model_validate(raw_json)

            # Filter out hallucinations not in the entity_names list
            final_triplets = []

            for triplet in validated_data.triplets:
                entity_subject = triplet.subject_entity
                entity_object = triplet.object_entity
                predicate_name = triplet.predicate.name

                # Check if subject/object are in the text but missed by the NER list
                is_valid_subject =\
                    entity_subject in entity_names or (entity_subject in source_text and len(entity_subject) > 2)
                is_valid_object =\
                    entity_object in entity_names or (entity_object in source_text and len(entity_object) > 2)

                if is_valid_subject and is_valid_object:

                    # Catch LLM blunder and enforce Directionality: Disease -> TREATMENT_OPTION -> Drug
                    if predicate_name == "TREATMENT_OPTION" and ("Infection" in entity_object or "Syndrome" in entity_object):
                        entity_subject, entity_object = entity_object, entity_subject

                    final_triplets.append({
                        "subject": entity_subject,
                        "predicate": predicate_name,
                        "object": entity_object
                    })
                else:
                    print(f"DEBUG: Dropping hallucinated entity in triplet: {triplet}")

            return final_triplets

        except Exception as e:
            print(f"Validation Error in Triplets: {e}")
            return []
