"""
Lightweight text dataset handler.

Provides utilities to:
- collect raw text samples with optional metadata,
- load datasets from JSONL files in a memory‑efficient streaming manner,
- enforce minimum text length and uniqueness constraints,
- manage dataset lifecycle (clear, status checks).
"""

from typing import Optional

from clusterer.utils import (
    get_logger,
    exists_input_file,
    load_jsonl_file_yield,
)


class TextDatasetHandler:
    """
    Container and loader for textual datasets with metadata.

    Attributes
    ----------
    name : str | None
        Optional dataset name (useful for logging/debug).
    dataset : list[str]
        Collected text samples.
    metadata : list[dict]
        Per-sample metadata aligned with `dataset`.
    min_text_len : int
        Minimum length of a text to be accepted during loading.
    full_debug : bool
        When True, logs each added sample at DEBUG level.
    """

    ACCEPT_INPUT_TYPES = ["jsonl"]

    def __init__(
        self,
        name: str | None = None,
        min_text_len: int = 30,
        full_debug: bool = False,
    ):
        """
        Initialize an empty dataset handler.

        Parameters
        ----------
        name : str | None, optional
            Human-readable dataset identifier.
        min_text_len : int, optional
            Minimum allowed text length for ingestion (default 30).
        full_debug : bool, optional (default False)
            If True, log each sample addition at DEBUG verbosity.
        """
        self.name = name
        self.dataset = []
        self.metadata = []

        self.min_text_len = min_text_len

        self.full_debug = full_debug

        self._logger = get_logger()

    def is_loaded(self):
        """
        Return True if any samples are currently loaded.
        """
        return len(self.dataset) > 0

    def add_example(self, example, example_metadata: Optional[dict] = None):
        """
        Append a single example and optional metadata.

        Parameters
        ----------
        example : Any
            Text content to store.
        example_metadata : dict | None, optional
            Per-sample metadata; if None, an empty dict is stored.
        """
        if self.full_debug:
            self._logger.debug(f"Adding example to dataset: {example}")

        self.dataset.append(example)
        self.metadata.append({} if example_metadata is None else example_metadata)

    def clear(self):
        """
        Remove all stored samples and metadata, resetting the handler.
        """
        self._logger.info("Clearing dataset")
        if self.dataset is not None:
            del self.dataset
            self._logger.info("  -> deleted dataset")
        if self.metadata is not None:
            self._logger.info("  -> deleted metadata")
            del self.metadata
        self.dataset = []
        self.metadata = []

        self._logger.info(
            f"Dataset size={len(self.dataset)} metadata size={len(self.metadata)}"
        )

    def load(
        self,
        file_path,
        text_column: str,
        metadata_column: Optional[str] = None,
        input_type: str = "jsonl",
        clear_dataset_if_exists: bool = True,
        only_unique_texts: bool = True,
    ) -> bool:
        """
        Load texts (and optional metadata) from a file.

        Parameters
        ----------
        file_path : str
            Path to the input file.
        text_column : str
            Key/column name containing the raw text.
        metadata_column : str | None, optional
            Key/column name for metadata in each record.
        input_type : str, optional
            Expected input format ("jsonl" supported).
        clear_dataset_if_exists : bool, optional (default True)
            If True, clears any existing samples before loading.
        only_unique_texts : bool, optional (default True)
            If True, skip records whose text duplicates a previously stored one.

        Returns
        -------
        bool
            True when loading succeeds; False otherwise.
        """
        if not self.__proper_input_type(input_type=input_type):
            self._logger.error(f"Input type {input_type} is not supported.")
            return False

        if not exists_input_file(file_path=file_path):
            self._logger.error(f"Input file {file_path} does not exist.")
            return False

        if clear_dataset_if_exists:
            self.dataset.clear()

        self._logger.info(f"Loading [{input_type}] dataset from {file_path}")
        if input_type == "jsonl":
            self.__load_jsonl_dataset(
                file_path=file_path,
                text_column=text_column,
                metadata_column=metadata_column,
                only_unique_texts=only_unique_texts,
            )
        else:
            raise Exception("Unsupported input type")

        return True

    def run(self):
        """Placeholder for future dataset processing hooks."""
        pass

    def __load_jsonl_dataset(
        self,
        file_path: str,
        text_column: str,
        metadata_column: Optional[str] = None,
        only_unique_texts: bool = True,
    ):
        """
        Stream and ingest records from a JSONL file.

        For each line:
        - ignores malformed JSON rows,
        - extracts `text_column` and optional `metadata_column`,
        - enforces `min_text_len`,
        - optionally skips duplicates when `only_unique_texts` is True.
        """
        if metadata_column is None:
            metadata_column = ""

        line_number = 0
        for line in load_jsonl_file_yield(file_path=file_path):
            line_number += 1
            if line is None:
                self._logger.warning(
                    f"  -> cannot load data from {line_number} line [{file_path}]"
                )
            data_str = line.get(text_column, None)
            if data_str is None:
                self._logger.warning(
                    f"Cannot find text column name'{text_column}' "
                    f"in line {line_number} -- skipping line."
                )
                continue

            data_str = data_str.strip()
            if len(data_str) < self.min_text_len:
                self._logger.info(
                    f"Skipping line {line_number} because of too short text"
                )

            if only_unique_texts and data_str in self.dataset:
                self._logger.info(
                    f"Skipping line {line_number} because of duplicated text"
                )
                continue

            metadata = line.get(metadata_column, {})
            self.add_example(example=data_str, example_metadata=metadata)

        self._logger.info(f" * Number of read examples: {line_number}")
        self._logger.info(f" * Number of loaded examples: {len(self.dataset)}")

    def __proper_input_type(self, input_type: str):
        """
        Return True if the input type is currently supported.
        """
        return input_type in self.ACCEPT_INPUT_TYPES
