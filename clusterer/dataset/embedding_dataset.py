"""
Embedding utilities based on Sentence-Transformers.

Defines a small hierarchy to:
- load and manage an embedding model/tokenizer,
- normalize and batch-encode text into vectors,
- pre-truncate texts to a safe token length for the model,
- optionally store original texts and produced vectors.
"""

import abc

from tqdm import tqdm
from typing import List
from numpy import ndarray

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from clusterer.utils import get_logger


class AnyVectorizedDataHandler(abc.ABC):
    """
    Abstract base for vectorized data handlers.

    Holds shared state such as the model, tokenizer, device, and buffers
    for produced vectors and raw text. Subclasses must implement model
    loading, text preprocessing, and vectorization.
    """

    def __init__(self, model_path: str, model_input_size: int, device: str):
        self.model = None
        self.tokenizer = None

        self.device = device
        self.model_path = model_path
        self.model_input_size = model_input_size

        self.vectors = []
        self.texts_str = []

        self._logger = get_logger()

    @abc.abstractmethod
    def load_model(self):
        """Load underlying embedding model and tokenizer."""
        raise NotImplementedError

    @abc.abstractmethod
    def vectorize(
        self,
        texts: list[str],
        store_vector: bool = True,
        store_text_str: bool = True,
        show_progress: bool = False,
    ):
        """
        Convert a list of texts into embeddings.

        Parameters
        ----------
        texts : list[str]
            Input texts to encode.
        store_vector : bool, optional (default: True)
            If True, cache produced vectors in self.vectors.
        store_text_str : bool, optional (default: True)
            If True, cache original texts in self.texts_str.
        show_progress : bool, optional (default: False)
            If True, show a progress bar during processing.

        Returns
        -------
        list[Any]
            Encoded vectors. Concrete subtype decides the element type.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def convert_to_proper_dataset(
        self, texts: list[str], show_progress: bool = False
    ):
        """
        Prepare texts for the model (e.g., truncate to max tokens).

        Parameters
        ----------
        texts : list[str]
            Raw texts to standardize/trim.
        show_progress : bool, optional
            If True, display progress.

        Returns
        -------
        list[str]
            Texts transformed to comply with model-input constraints.
        """
        raise NotImplementedError


class SentenceTransformerEmbedding(AnyVectorizedDataHandler, abc.ABC):
    """
    Implementation for Sentence-Transformers encoders.
    """

    def __init__(self, model_path: str, model_input_size: int, device: str):
        super().__init__(
            model_path=model_path, model_input_size=model_input_size, device=device
        )

        self.max_tokens_in_text = self.model_input_size - 4

    def load_model(self):
        """
        Load a SentenceTransformer model and its tokenizer and move to device if needed.
        """
        assert self.model_path is not None
        assert len(self.model_path) > 0

        self._logger.info(f"Loading model from {self.model_path}")
        self.model = SentenceTransformer(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        if "cuda" in self.device:
            self._logger.info(f"Moving model to device: {self.device}")
            self.model = self.model.to(self.device)

    def convert_to_proper_dataset(
        self, texts: List[str], show_progress: bool = False
    ):
        """
        Trim texts to respect the maximum token length for the model.

        Tokens beyond `self.max_tokens_in_text` are removed, then decoded
        back to text to keep alignment with the tokenizer's expectations.

        Parameters
        ----------
        texts : list[str]
            Raw texts to inspect and truncate if necessary.
        show_progress : bool, optional
            If True, display a progress bar.

        Returns
        -------
        list[str]
            Texts guaranteed not to exceed the token budget.
        """

        pbar = None
        if show_progress:
            pbar = tqdm(total=len(texts), desc="Converting texts to proper dataset")

        proper_length_data = []
        for orig_text in texts:
            text_tokens = self.tokenizer(orig_text).input_ids
            if len(text_tokens) > self.max_tokens_in_text:
                proper_text = self.tokenizer.decode(
                    token_ids=text_tokens[: self.max_tokens_in_text],
                    skip_special_tokens=True,
                )
                proper_length_data.append(proper_text)
            else:
                proper_length_data.append(orig_text)

            if pbar is not None:
                pbar.update(1)
        return proper_length_data

    def convert_to_embeddings(
        self, texts: List[str], normalize_embeddings: bool = True
    ) -> ndarray:
        """
        Encode texts into embeddings using Sentence-Transformers.

        Parameters
        ----------
        texts : list[str]
            Preprocessed texts to encode.
        normalize_embeddings : bool, optional (default: True)
            If True, return L2-normalized vectors from the model.

        Returns
        -------
        numpy.ndarray
            A matrix of shape (n_texts, embedding_dim).
        """

        embeddings = self.model.encode(
            texts, normalize_embeddings=normalize_embeddings
        )
        assert len(embeddings) == len(texts)
        return embeddings


class EmbeddingsDatasetHandler(SentenceTransformerEmbedding):
    """
    High-level handler to vectorize text collections and manage caching.
    """

    def __init__(
        self,
        embedder_path: str,
        embedder_input_size: int,
        device: str,
        load_model: bool = True,
    ):
        """
        Configure the handler and optionally load the model.

        Parameters
        ----------
        embedder_path : str
            Identifier or path for the Sentence-Transformers model.
        embedder_input_size : int
            Maximum token length the model should accept.
        device : str
            Target device spec (e.g., "cpu", "cuda", "cuda:0").
        load_model : bool, optional (default: True)
            If True, load the model immediately.
        """
        super().__init__(
            model_path=embedder_path,
            model_input_size=embedder_input_size,
            device=device,
        )
        if load_model and len(self.model_path):
            self.load_model()

    def clear(self):
        """
        Clear cached vectors and texts.
        """
        self.vectors.clear()
        self.texts_str.clear()

    def vectorize(
        self,
        texts: list[str],
        store_vector: bool = True,
        store_text_str: bool = True,
        show_progress: bool = False,
        normalize_embeddings: bool = True,
    ):
        """
        Vectorize a list of texts end-to-end.

        Steps:
        1) Optionally cache the original texts.
        2) Truncate/normalize texts to model limits.
        3) Batch-encode texts to embeddings (optionally normalized).
        4) Optionally cache computed embeddings.

        Parameters
        ----------
        texts : list[str]
            Texts to encode.
        store_vector : bool, optional
            If True, cache embeddings in `self.vectors`.
        store_text_str : bool, optional (default: True)
            If True, cache original texts in `self.texts_str`.
        show_progress : bool, optional (default: False)
            If True, show progress bars for preprocessing and encoding.
        normalize_embeddings : bool, optional
            If True, return normalized vectors.

        Returns
        -------
        list[Any]
            The produced embeddings (also cached if requested).
        """

        if not len(texts):
            return []

        if store_text_str:
            self.texts_str = texts

        proper_texts = self.convert_to_proper_dataset(
            texts=texts, show_progress=show_progress
        )

        return self._convert_to_embeddings(
            texts=proper_texts,
            store_vector=store_vector,
            show_progress=show_progress,
            normalize_embeddings=normalize_embeddings,
        )

    def _convert_to_embeddings(
        self,
        texts: List[str],
        store_vector: bool = True,
        show_progress: bool = False,
        normalize_embeddings: bool = True,
    ):
        """
        Batch-encode texts to embeddings and optionally cache them.

        Parameters
        ----------
        texts : list[str]
            Preprocessed texts to encode.
        store_vector : bool, optional (default: True)
            If True, store embeddings to `self.vectors`.
        show_progress : bool, optional (default: False)
            If True, show progress for batches.
        normalize_embeddings : bool, optional (default: True)
            If True, return normalized vectors.

        Returns
        -------
        list[Any]
            Sequence of embeddings in the same order as inputs.
        """

        pbar = None
        if show_progress:
            pbar = tqdm(total=len(texts), desc="Converting texts to embeddings")

        texts_embeddings = []
        for batch in self._batched_dataset(batch_size=500, data=texts):
            embeddings = self.convert_to_embeddings(
                texts=batch, normalize_embeddings=normalize_embeddings
            )
            texts_embeddings.extend(embeddings)
            if pbar is not None:
                pbar.update(len(batch))

        if store_vector:
            self.vectors = texts_embeddings
        return texts_embeddings

    @staticmethod
    def _batched_dataset(batch_size: int, data: List[str]):
        """
        Yield successive batches of size `batch_size` from `data`.
        """
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]
