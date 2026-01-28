import csv
import json
from pathlib import Path


def log_triplets(triplets: list[dict], filename: str = "triplets.csv") -> None:
    """
    Export the triplets to a CSV file.

    Args:
        triplets: List of dictionaries of type [{'subject': 'Entity A', 'predicate': 'Predicate', 'object': 'Entity B'}, ...].
        filename: Target filename.
    """
    if not triplets:
        print("Warning: No triplets to log.")
        return

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    csv_headers = ["subject", "predicate", "object", "source"]

    filepath = log_dir / filename
    file_exists = filepath.exists() and filepath.stat().st_size > 0

    try:
        with open(filepath, "w", newline="", encoding="utf-8") as fid:
            csv_writer = csv.DictWriter(fid, fieldnames=csv_headers, extrasaction='ignore')
            csv_writer.writeheader()
            csv_writer.writerows(triplets)

        print(f"✅ Successfully logged {len(triplets)} triplets to {filepath}.")

    except Exception as e:
        print(f"❌ Error logging triplets: {e}")


def log_canonical_entities(entities: list[dict]) -> None:
    """
    Export the canonical entities metadata to a CSV file.

    Args:
        entities: Dictionary storing all canonical entities, e.g., {"Hypertension": {"name": "Hypertension", "type": "Disease"}}.
        filename: Target filename.
    """
    if not entities:
        print("Warning: No entities found in registry.")
        return

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    all_keys = set()
    for entry in entities:
        all_keys.update(entry.keys())

    # Sort headers, while ensuring "name" is the first column for readability
    sorted_keys = sorted(list(all_keys))
    if "name" in sorted_keys:
        sorted_keys.insert(0, sorted_keys.pop(sorted_keys.index("name")))

    filename = "canonical_entities.csv"
    filepath = log_dir / filename

    try:
        with open(filepath, "w", newline="", encoding="utf-8") as fid:
            csv_writer = csv.DictWriter(fid, fieldnames=sorted_keys, restval="")
            csv_writer.writeheader()
            csv_writer.writerows(entities)

        print(f"✅ Successfully logged {len(entities)} entities to {filepath}.")

    except Exception as e:
        print(f"❌ Error logging canonical entities: {e}")


def log_chunks(chunks: list[dict]) -> None:
    """
    Export the chunks to a text file for visual inspection.

    Args:
        chunks: List of dictionaries representing processed document segments.
    """

    if not chunks:
        print("Warning: No chunks to log.")
        return

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    source_name = chunks[0].get("source", "unknown_source")
    source_name_clean = "".join([c for c in source_name if c.isalnum() or c in (' ', '.', '_')]).rstrip()
    filename = f"chunks_{source_name_clean}.json"

    filepath = log_dir / filename

    try:
        with open(filepath, "w", newline="", encoding="utf-8") as fid:
            for chunk in chunks:
                json_record = json.dumps(chunk, ensure_ascii=False)
                fid.write(json_record + "\n")

        print(f"✅ Successfully logged {len(chunks)} chunks to {filepath}.")

    except Exception as e:
        print(f"❌ Error logging chunks: {e}")


def log_aliases(alias_lookup_map: dict[str, str]) -> None:
    """
    Export the alias lookup map to a JSON file for visual inspection.

    Args:
        alias_lookup_map: Dictionary mapping synonyms to canonical names.
    """
    if not alias_lookup_map:
        print("Warning: No aliases to log.")
        return

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    filename = "alias_synonym_lookup.json"
    filepath = log_dir / filename

    try:
        with open(filepath, "w", encoding="utf-8") as fid:
            json.dump(alias_lookup_map, fid, ensure_ascii=False, indent=4)

        print(f"✅ Successfully logged {len(alias_lookup_map)} alias mappings to {filepath}.")
    except Exception as e:
        print(f"❌ Error logging aliases: {e}")
