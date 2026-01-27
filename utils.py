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

    csv_headers = ["subject", "predicate", "object"]

    filepath = log_dir / filename
    file_exists = filepath.exists() and filepath.stat().st_size > 0
    with open(filepath, "a", newline="", encoding="utf-8") as fid:
        csv_writer = csv.DictWriter(fid, fieldnames=csv_headers)

        if not file_exists:
            csv_writer.writeheader()

        csv_writer.writerows(triplets)

    print(f"Successfully logged {len(triplets)} triplets to {filepath}.")


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

    with open(filepath, "w", newline="", encoding="utf-8") as fid:
        for chunk in chunks:
            json_record = json.dumps(chunk, ensure_ascii=False)
            fid.write(json_record + "\n")

    print(f"Successfully logged {len(chunks)} chunks to {filepath}.")
