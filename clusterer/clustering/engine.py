"""
Clustering engine abstraction.

Provides a unified interface for running different clustering backends over
a feature matrix. Currently, supports HDBSCAN and exposes a small API to fit
a model and retrieve labels while keeping track of the parameters used.
"""

import hdbscan

from typing import Optional, List

from clusterer.utils import get_logger


class ClustererEngine:
    """
    Thin wrapper around concrete clustering implementations.

    This class dispatches to a selected clustering method (e.g., HDBSCAN),
    stores the fitted model and produced labels, and keeps the parameters
    associated with the latest run.

    Attributes
    ----------
    method : str
        Name of the clustering method to use.
    params : dict | None
        Parameters used for the most recent clustering run.
    clustering_model : Any | None
        The fitted clustering model instance, when available.
    labels : list[int]
        Labels produced by the most recent clustering run.
    """

    AVAILABLE_METHODS = ["hdbscan"]

    def __init__(self, method: str, clustering_params: Optional[dict]):
        """
        Initialize the engine with a method and optional default parameters.

        Parameters
        ----------
        method : str
            Clustering backend to use. Must be in `AVAILABLE_METHODS`.
        clustering_params : dict | None
            Default parameters to use if per-call options are not supplied.

        Raises
        ------
        AssertionError -- If the provided method is not supported.
        """
        self.method = method
        assert (
            self.method in self.AVAILABLE_METHODS
        ), f"Unknown method: {self.method}"

        self.params = clustering_params

        self.clustering_model = None
        self.labels = []

        self._logger = get_logger()

    def prepare_clusters(
        self,
        dataset,
        options: Optional[dict] = None,
        debug: bool = False,
    ) -> Optional[List[int]]:
        """
        Fit the configured clustering model and return labels.

        Parameters
        ----------
        dataset : array-like
            Feature matrix to cluster. Must be compatible with the chosen backend.
        options : dict | None, optional
            Parameters for this run; overrides the engine's default params.
        debug : bool, optional,
            If True, logs additional information.

        Returns
        -------
        list[int] | None
            Cluster labels for each sample, or None on failure.
        """
        if debug:
            self._logger.info(f"Preparing clusters with {self.method} method")
        if self.method == "hdbscan":
            return self.__prepare_hdbscan_clusters(
                dataset=dataset, options=options, debug=debug
            )

        self._logger.error(f"Not support clustering method {self.method}")
        return None

    def __prepare_hdbscan_clusters(
        self,
        dataset,
        options: Optional[dict] = None,
        debug: bool = False,
    ) -> Optional[list[int]]:
        """
        Run clustering using HDBSCAN.

        Parameters
        ----------
        dataset : array-like
            Feature matrix to cluster.
        options : dict | None, optional
            HDBSCAN initialization parameters for this run. If None, uses
            the engine's stored params.
        debug : bool, optional,
            If True, logs parameter details.

        Returns
        -------
        list[int] | None
            Labels produced by HDBSCAN, or None if parameters
            are missing or fitting fails.
        """
        if self.clustering_model is not None:
            del self.clustering_model
            self.clustering_model = None

        clustering_params = options if options is not None else self.params
        if clustering_params is None or not len(clustering_params):
            self._logger.error("Clustering parameters are not set.")
            return None

        if debug:
            self._logger.info(f"Clustering parameters: {clustering_params}")

        self.clustering_model = hdbscan.HDBSCAN(**clustering_params).fit(dataset)
        if not self.clustering_model:
            return None

        self.labels = self.clustering_model.labels_
        self.params = clustering_params

        return self.labels
