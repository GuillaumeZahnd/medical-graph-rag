import os
import csv
from knowledge_graph_ingestor import KnowledgeGraphIngestor


class EntityRegistry:
    def __init__(self, ingestion_engine: KnowledgeGraphIngestor) -> None:
        self.ingestion_engine = ingestion_engine

        # Translator: Dictionary that maps every messy name variation or synonym found in the text to a unique canonical name.
        # Example: {"hta": "Hypertension", "high bp": "Hypertension", "High Blood Pressure", "Hypertension", ...}
        self._synonym_lookup_map: dict[str, str] = {}

        # Store: Dictionary that stores the rich metadata for the entities that will constitute the nodes of the graph.
        # Example: {"Hypertension": {"name": "Hypertension", "type": "Disease"}}
        self._canonical_entity_store: dict[str, dict] = {}


    def update_canonical_entities(self, raw_entities_batch: list[dict]) -> None:
        """
        Mutate class state by resolving unknown entities via LLM.
        """
        # Determine whether these entities are already resolved, otherwise we will resolved them via LLM.
        # Use a dictionary to de-duplicate within the batch itself to save tokens.
        unique_unresolved_entities = {}
        for raw_entity in raw_entities_batch:
            raw_name = raw_entity.get("name")
            key = raw_name.lower().strip()
            if key not in self._synonym_lookup_map and key not in unique_unresolved_entities:
                unique_unresolved_entities[key] = raw_entity

        unresolved_entities = list(unique_unresolved_entities.values())
        if not unresolved_entities:
            return

        sub_batch_size = 10
        for i in range(0, len(unresolved_entities), sub_batch_size):
            sub_list = unresolved_entities[i:i + sub_batch_size]

            # Use the LLM of the ingestion engine to resolve the unresolved entities, that is, the "newly discovered" names.
            resolved_entities = self.ingestion_engine.resolve_entities(entities=sub_list)

            if len(sub_list) != len(resolved_entities):
                print(f"CRITICAL: Sent {len(sub_list)} entities to the LLM, received {len(resolved_entities)}.")
                continue

            for raw, normalized in zip(sub_list, resolved_entities, strict=True):
                canonical_name = normalized.get("name") if isinstance(normalized, dict) else None

                if not isinstance(canonical_name, str) or not canonical_name.strip():
                    continue

                raw_name = raw.get("name")
                if not raw_name:
                    continue

                # Add an entry in the dictionary to map the raw name to the canonical name.
                raw_key = raw_name.lower().strip()
                canonical_key = canonical_name.lower().strip()

                # Map the synonym.
                self._synonym_lookup_map[raw_key] = canonical_name

                # Map the canonical name to itself (avoids future LLM roundtrips).
                if canonical_key not in self._synonym_lookup_map:
                    self._synonym_lookup_map[canonical_key] = canonical_name

                # Store metadata only once per unique canonical entity.
                if canonical_name not in self._canonical_entity_store:
                    self._canonical_entity_store[canonical_name] = normalized


    @property
    def synonym_lookup_map(self) -> dict[str, str]:
        """
        Return a copy of the synonym-to-canonical mapping.
        """
        return self._synonym_lookup_map.copy()


    @property
    def canonical_entity_store(self) -> list[dict]:
        """
        Return the list of unique, resolved medical entities with metadata.
        """
        return list(self._canonical_entity_store.values())


    @property
    def entity_count(self) -> int:
        """
        Return the number of unique canonical entities stored.
        """
        return len(self._canonical_entity_store)


    @property
    def synonym_count(self) -> int:
        """
        Return the total number of variations/synonyms mapped.
        """
        return len(self._synonym_lookup_map)


    def load_canonical_entities(self, csv_path: str, csv_name: str) -> None:
        """
        Update the class state by loading a precomputed CSV file of entities instead exploring and resolving text chunks.
        Expects a CSV where one column is 'name' and another is 'type'.
        """
        csv_path_and_name = os.path.join(csv_path, csv_name)
        if not os.path.exists(csv_path_and_name):
            raise FileNotFoundError(f"Provided CSV file not found: {csv_path_and_name}")

        count = 0

        with open(csv_path_and_name, mode='r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                canonical_name = row.get("name")
                if not canonical_name:
                    continue

                self._canonical_entity_store[canonical_name] = dict(row)

                canonical_key = canonical_name.lower().strip()
                self._synonym_lookup_map[canonical_key] = canonical_name

                count += 1

        print(f"Successfully loaded {count} canonical entities from {csv_name}.")


    def get_canonical_names_from_raw(self, raw_entity_names: list[str]) -> list[str]:
        """
        Convert a list of raw entity names found in a chunk into their canonical versions.
        """
        results = set()
        for name in raw_entity_names:
            key = name.lower().strip()
            canonical = self._synonym_lookup_map.get(key, name)
            results.add(canonical)
        return list(results)


    def log_entities(self, filename: str) -> None:
        """
        Export the canonical entities metadata to a CSV file.
        """
        entities = self.canonical_entity_store
        if not entities:
            print("Warning: No entities found in registry.")
            return

        path_to_results = "logs"
        os.makedirs(path_to_results, exist_ok=True)

        all_keys = set()
        for e in entities:
            all_keys.update(e.keys())
        csv_headers = sorted(list(all_keys))

        filepath = os.path.join(path_to_results, f"{filename}")
        with open(filepath, "w", newline="", encoding="utf-8") as fid:
            csv_writer = csv.DictWriter(fid, fieldnames=csv_headers, restval="")
            csv_writer.writeheader()
            csv_writer.writerows(entities)

        print(f"Successfully logged {len(entities)} entities to {filepath}.")

