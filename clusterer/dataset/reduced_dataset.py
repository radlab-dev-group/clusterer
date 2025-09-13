"""Dimensionality reduction helpers for clustering.

This module provides a handler that transforms high-dimensional embeddings
into f.e. 2D representations using either t-SNE or UMAP, configurable with
optimizer and similarity metric options.
"""

from umap import UMAP
from typing import Optional
from sklearn.manifold import TSNE


class ReducedClusteringDatasetHandler:
    """
    Manage dimensionality reduction prior to clustering.

    Supports the following reducers:
    - t-SNE (via sklearn.manifold.TSNE)
    - UMAP (via umap-learn)

    Attributes
    ----------
    method : str
        Reduction method name ("tsne" or "umap").
    optim : str | None
        Optimizer variant for t-SNE ("barnes_hut" or "exact").
    sim_metric : str | None
        Similarity metric used by the reducer (e.g., "cosine").
    dataset : list | array-like
        Stores the most recent reduced dataset if requested.
    workdir : str | None
        Optional working directory for future extensions.
    """

    AVAILABLE_REDUCERS = ["tsne", "umap"]
    AVAILABLE_OPTIMIZERS = ["barnes_hut", "exact"]

    def __init__(
        self,
        method: str,
        workdir: Optional[str] = None,
        optim: Optional[str] = None,
        sim_metric: Optional[str] = None,
    ):
        """
        Initialize the handler with a reducer configuration.

        Parameters
        ----------
        method : str
            Reduction algorithm to use ("tsne" or "umap").
        workdir : str | None, optional (default None)
            Optional working directory (reserved for future use).
        optim : str | None, optional (default None)
            Optimizer for t-SNE ("barnes_hut" or "exact").
        sim_metric : str | None, optional (default None)
            Similarity/distance metric (e.g., "cosine").
        """
        self.method = method
        self.optim = optim
        self.sim_metric = sim_metric

        assert (
            self.method in self.AVAILABLE_REDUCERS
        ), f"Unknown method: {self.method}"
        assert (
            self.optim in self.AVAILABLE_OPTIMIZERS
        ), f"Unknown optimizer: {self.optim}"

        self.dataset = []
        self.workdir = workdir

    def reduce_dataset(
        self, embeddings, store_vectors: bool = True, reduced_dim: int = 2
    ):
        """
        Reduce embeddings to a `reduced_dim` space using the configured reducer.

        Parameters
        ----------
        embeddings : array-like
            High-dimensional feature vectors to be reduced.
        store_vectors : bool, optional
            If True, stores the reduced representation in `self.dataset`.
        reduced_dim : int, optional (default 2)
            Dimensionality of reduced embedding.

        Returns
        -------
        array-like
            The reduced to `reduced_dim` embedding matrix.
        """
        if self.method == "tsne":
            red_dataset = self.__prepare_to_clustering_tsne(
                embeddings=embeddings, reduced_dim=reduced_dim
            )
        elif self.method == "umap":
            red_dataset = self.__prepare_to_clustering_umap(
                embeddings, reduced_dim=reduced_dim
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        if store_vectors:
            self.dataset = red_dataset

        return red_dataset

    def __prepare_to_clustering_tsne(self, embeddings, reduced_dim: int):
        """
        Apply t-SNE to produce a `reduced_dim` projection of the embeddings.

        Parameters
        ----------
        embeddings : array-like
            Input feature matrix.

        Returns
        -------
        array-like
            `reduced_dim` coordinates produced by t-SNE.
        """
        conv_embeddings = TSNE(
            n_components=reduced_dim,
            learning_rate="auto",
            perplexity=30,
            method=self.optim,
            metric=self.sim_metric,
        ).fit_transform(embeddings)
        return conv_embeddings

    def __prepare_to_clustering_umap(self, embeddings, reduced_dim: int):
        """
        Apply UMAP to produce a `reduced_dim` projection of the embeddings.

        Parameters
        ----------
        embeddings : array-like
            Input feature matrix.

        Returns
        -------
        array-like
            `reduced_dim` coordinates produced by UMAP.
        """

        conv_embeddings = UMAP(
            n_components=reduced_dim, metric=self.sim_metric
        ).fit_transform(embeddings)
        return conv_embeddings
