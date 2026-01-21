import os
from pathlib import Path


class Dataset:
    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = dataset_path
        self._pdf_collection = []


    def populate_collection(self) -> None:
        """
        Populate the PDF collection by constructing a list of tuples (full_path, file_name).
        """

        root_path = Path(self.dataset_path)

        for child in root_path.rglob("*"):
            if child.is_file() and child.suffix.lower() == ".pdf":
                relative_parent = child.parent.relative_to(root_path)
                full_path = root_path / relative_parent
                self._pdf_collection.append((full_path.as_posix(), child.name))


    @property
    def pdf_collection() -> list:
        """
        Return the PDF collection.
        """
        return self._pdf_collection


    def display_collection(self) -> None:
        """
        Display the PDF collection.
        """
        for folder, file in self._pdf_collection:
            print(f"{file:60} | {folder}")
