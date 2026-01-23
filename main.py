import os
from mistralai import Mistral

from dataset import Dataset
from entity_registry import EntityRegistry
from ingestion_engine import IngestionEngine
from document_extractor import DocumentExtractor


if __name__ == "__main__":

    client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
    model = "mistral-large-latest"

    dataset = Dataset(dataset_path="mini_dataset")
    dataset.populate_collection()
    dataset.display_collection()

    extractor = DocumentExtractor(max_chunk_size=2000)
    ingestion_engine = IngestionEngine(client=client, model=model)
    entity_registry = EntityRegistry(ingestion_engine=ingestion_engine)

    # [0] Collect all text chunks from all documents
    chunks_collection = []
    for doc in dataset.pdf_collection:
        print(f"Collecting chunks from {doc['file_name']}...")
        chunks = extractor.process_pdf(
            pdf_path=doc["absolute_path"], pdf_name=doc["file_name"], hierarchy=doc["hierarchy"], priority=1.0)
        extractor.log_chunks(processed_data=chunks)
        chunks_collection.extend(chunks)

    nb_chunks = len(chunks_collection)

    batch_size = 5

    # Possibility to completely bypass the first pass to save time
    rebuild_canonical_entities = False

    if rebuild_canonical_entities:

        print(f"\n\nFIRST PASS: ENTITY EXPLORATION AND RESOLUTION\nNUMBER OF CHUNKS: {nb_chunks}")
        for i in range(0, nb_chunks, batch_size):
            print(f"Processing chunks [{i}-{i + batch_size - 1}]...")

            batch = chunks_collection[i : i + batch_size]
            batch_text = [c['data'] for c in batch]

            # [1] Extract as many entity names as possible from the text chunk (high recall)
            batch_raw_entities = ingestion_engine.extract_entities(batch_text=batch_text)

            # [2] Detect whether there are new entities
            all_entities_in_this_batch = [ent for sublist in batch_raw_entities for ent in sublist]
            registry_keys = entity_registry.synonym_lookup_map
            new_entities = [
                e for e in all_entities_in_this_batch
                if e['name'].lower().strip() not in registry_keys
            ]

            # [3] Resolve new entities (either map them to an existing canonical form, or create a new canonical form)
            if new_entities:
                entity_registry.update_canonical_entities(new_entities)

        entity_registry.log_entities(filename="canonical_entities.csv")

    else:
        csv_path="logs"
        csv_name="canonical_entities.csv"
        entity_registry.load_canonical_entities(csv_path=csv_path, csv_name=csv_name)


    print(f"\n\nSECOND PASS: ENTITY-PREDICATE-ENTITY TRIPLET EXTRACTION \nNUMBER OF CHUNKS: {nb_chunks}")
    for i in range(0, nb_chunks, batch_size):
        print(f"Processing chunks [{i}-{i + batch_size - 1}]...")

        batch = chunks_collection[i : i + batch_size]
        batch_text = [c['data'] for c in batch]

        # [4] Extract directed semantic relationships between entities
        triplets = ingestion_engine.extract_triplets(
            source_text=batch_text, entities=entity_registry.canonical_entity_store)

