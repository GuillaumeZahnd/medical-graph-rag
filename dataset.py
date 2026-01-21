import os
from pathlib import Path


class Dataset:
    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = dataset_path
        self._pdf_collection = []


    def populate_collection(self) -> None:
        """
        Populates the PDF collection with physical paths and graph metadata.
        """
        root_path = Path(self.dataset_path)
        root_name = root_path.name

        for child in root_path.rglob("*"):
            if child.is_file() and child.suffix.lower() == ".pdf":
                relative_parent = child.parent.relative_to(root_path)
                hierarchy = (Path(root_name) / relative_parent).as_posix()
                absolute_path = str(child.absolute())
                full_path = Path(root_name) / relative_parent
                self._pdf_collection.append({
                    "absolute_path": absolute_path,
                    "hierarchy": hierarchy,
                    "file_name": child.name
                })


    @property
    def pdf_collection(self) -> list[dict]:
        """
        Return the PDF collection.
        """
        return self._pdf_collection


    @property
    def pdf_count(self) -> int:
        """
        Return the number of PDFs in the collection.
        """
        return len(self._pdf_collection)


    def display_collection(self) -> None:
        """
        Display the PDF collection.
        """
        print(f"NUMBER OF DOCUMENTS: {len(self._pdf_collection)}")
        print(f"{'FILENAME':<60} | {'GRAPH HIERARCHY'}")
        print("-" * 100)
        for item in self._pdf_collection:
            print(f"{item['file_name']:<60} | {item['hierarchy']}")
