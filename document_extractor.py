import os
import re
import uuid
import pymupdf4llm


class DocumentExtractor:

    # Define the atomic unit of vertical movement
    LINE_TERMINATOR = "\n"

    # Define the standard separation for semantic blocks (paragraphs/sections)
    STANDARD_BLOCK_SPACING = LINE_TERMINATOR * 2

    # Define the standard separation for inline elements (sentences/words)
    INLINE_SPACING = " "

    # Content of the first capturing group
    FIRST_GROUP_CONTENT = r"\1"

    # Catch excessive vertical whitespace, even if contaminated with tabs/spaces
    EXCESSIVE_NEWLINES_RE = re.compile(r"(?:\n[ \t]*){3,}")

    # Catch pagination artifacts (e.g., " 12 ", "- 1 -") that break flow
    PAGINATION_ARTIFACT_RE = re.compile(r"\n[ \t]*[\[\-\(]?\d{1,3}[\]\-\)]?[ \t]*\n")

    # Match horizontal whitespace at the start or end of any line
    LINE_FRINGE_WHITESPACE_RE = re.compile(r"^[ \t]+|[ \t]+$", re.MULTILINE)

    # Identify structural boundaries defined by level-2 headers
    SECTION_BOUNDARY_RE = re.compile(r"(^##\s.*)", re.MULTILINE)

    # Match punctuation followed by space, avoiding decimals
    SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?]) +')

    # Match an underscore, capture everything inside that isn't an underscore, match closing underscore
    ITALIC_CLEANUP_RE = re.compile(r'_([^_]+)_')


    def __init__(self, max_chunk_size: int) -> None:
        self.max_chunk_size = max_chunk_size


    def polish_markdown(self, text: str) -> str:
        """
        Apply minor cleaning operations to the text.

        Args:
            text: Raw markdown string.

        Returns:
            Clean markdown string.
        """
        # Remove pagination noise
        text = self.PAGINATION_ARTIFACT_RE.sub(self.LINE_TERMINATOR, text)

        # Clean up "fringe" whitespace (spaces at start/end of lines)
        text = self.LINE_FRINGE_WHITESPACE_RE.sub("", text)

        # Remove italic underscore tags
        text = self.ITALIC_CLEANUP_RE.sub(self.FIRST_GROUP_CONTENT, text)

        # Normalize the spacing between blocks
        text = self.EXCESSIVE_NEWLINES_RE.sub(self.STANDARD_BLOCK_SPACING, text)

        text = text.strip()

        return text


    def semantic_chunking(self, text: str) -> list[str]:
        """
        Slice the text into smaller chunks.

        Args:
            text: Markdown string.

        Returns:
            List of chunks.
        """
        # Split by secondary headers (##)
        sections = self.SECTION_BOUNDARY_RE.split(text)

        # If no headers found, treat the text as one (potentially large) section
        if len(sections) == 1:
            return self._handle_oversized_chunk(sections[0])

        chunks = []

        # Handle index 0 (content before the first ##)
        if sections[0].strip():
            chunks.extend(self._handle_oversized_chunk(text=sections[0].strip()))

        # Re-combine headers with their following content
        for i in range(1, len(sections), 2):
            header = sections[i].strip()
            content = sections[i+1].strip() if i+1 < len(sections) else ""
            combined = f"{header}\n{content}"
            chunks.extend(self._handle_oversized_chunk(combined))

        return [c for c in chunks if len(c) > 50]


    def _handle_oversized_chunk(self, text: str) -> list[str]:
        """
        Helper to split potentially large text blocks by paragraphs.

        Args:
            text: Block of text.

        Returns:
            List of strings, each within the maximal chunk size limit.
        """
        if len(text) <= self.max_chunk_size:
            return [text]

        # Try splitting by paragraphs (Primary)
        if self.STANDARD_BLOCK_SPACING in text:
            return self._recombine_splits(
                fragments=text.split(self.STANDARD_BLOCK_SPACING),
                separator=self.STANDARD_BLOCK_SPACING
            )

        # Try splitting by sentences (Secondary)
        sentences = self.SENTENCE_SPLIT_RE.split(text)
        if len(sentences) > 1:
            return self._recombine_splits(
                fragments=sentences,
                separator=self.INLINE_SPACING
            )

        # Try splitting by words (Tertiary)
        words = text.split(" ")
        if len(words) > 1:
            return self._recombine_splits(
                fragments=words,
                separator=self.INLINE_SPACING
            )

        # Fallback: Hard character split (Last resort)
        return [text[i : i + self.max_chunk_size] for i in range(0, len(text), self.max_chunk_size)]


    def _recombine_splits(self, fragments: list[str], separator: str) -> list[str]:
        """
        Greedily recombine text fragments into chunks within a size limit. Iterates through fragments (paragraphs,
        sentences, or words) and joins them using the provided separator as long as the resulting chunk remains under
        the hard character limit.

        Args:
            fragments: List of text fragments to be merged.
            separator: String used to join fragments (e.g., "\n\n", " ", or "").

        Returns:
            List of strings where each element is maximally long without exceeding the character limit.
        """
        chunks = []
        current_batch = []
        current_len = 0
        sep_len = len(separator)

        for item in fragments:
            item_len = len(item)

            # If a single element is still too large, recurse further
            if item_len > self.max_chunk_size:
                if current_batch:
                    chunks.append(separator.join(current_batch))
                    current_batch = []
                    current_len = 0
                chunks.extend(self._handle_oversized_chunk(text=item))
                continue

            if current_len + item_len + (sep_len if current_batch else 0) <= self.max_chunk_size:
                current_batch.append(item)
                current_len += item_len + (sep_len if len(current_batch) > 1 else 0)
            else:
                chunks.append(separator.join(current_batch))
                current_batch = [item]
                current_len = item_len

        if current_batch:
            chunks.append(separator.join(current_batch))

        return chunks


    def process_pdf(self, pdf_path: str, pdf_name: str, hierarchy: str, priority: float) -> list[dict]:
        """
        Orchestrate the extraction and normalization of PDF content into database-ready chunks.

        Args:
            pdf_path: Filesystem path to the target PDF document.
            pdf_name: Original filename to be stored as the metadata source.
            hierarchy: Organizational path or folder structure context of the document.
            priority: Numerical weighting representing the document's authority or trust level.

        Returns:
            List of dictionaries, where each entry represents a semantic chunk enriched with a unique UUID
            and document metadata.
        """

        # Convert the PDF to Markdown
        markdown_raw = pymupdf4llm.to_markdown(pdf_path)

        # Apply minor cleaning
        markdown_clean = self.polish_markdown(text=markdown_raw)

        # Slice the text into smaller chunks
        chunks = self.semantic_chunking(markdown_clean)

        # Format for Neo4j database
        processed_data = []
        for chunk in chunks:
            processed_data.append({
                "id": str(uuid.uuid4()),
                "source": pdf_name,
                "hierarchy": hierarchy,
                "priority": priority,
                "data": chunk
            })

        return processed_data


    def log_chunks(self, processed_data: list[dict]) -> None:
        """
        Export the chunks to a text file for visual inspection.
        """

        if not processed_data:
            return

        path_to_results = "logs"
        os.makedirs(path_to_results, exist_ok=True)

        source_name = processed_data[0].get("source", "unknown_source")
        source_name_clean = "".join([c for c in source_name if c.isalnum() or c in (' ', '.', '_')]).rstrip()
        filepath = os.path.join(path_to_results, f"{source_name_clean}.txt")

        with open(filepath, "w", newline="", encoding="utf-8") as fid:
            for chunk_id, chunk_data in enumerate(processed_data):
                fid.write(f"Chunk index: {chunk_id}\n")
                for key, val in chunk_data.items():
                    fid.write(f"[{key}]: {val}\n")
                fid.write("-"*64 + "\n")
