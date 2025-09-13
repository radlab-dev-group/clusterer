"""
clusterer.clustering.clusterer

This module implements: class:`RdlClusterer`, a high‑level orchestrator that
loads raw text data, creates embeddings, optionally reduces dimensionality,
runs one or more clustering algorithms, and selects the best clustering
configuration based on configurable size constraints.
"""

import tqdm
import numpy as np

from typing import Optional, Dict, List

from clusterer.utils import get_logger
from clusterer.clustering.engine import ClustererEngine
from clusterer.dataset.text_dataset import TextDatasetHandler
from clusterer.dataset.embedding_dataset import EmbeddingsDatasetHandler
from clusterer.dataset.reduced_dataset import ReducedClusteringDatasetHandler


class RdlClusterer:
    """High‑level clustering pipeline for textual datasets.

    The class encapsulates the full workflow:

    1. Load raw text records via: class:`TextDatasetHandler`.
    2. Convert texts to embeddings using: class:`EmbeddingsDatasetHandler`.
    3. Optionally reduce the embedding dimensionality with
       class:`ReducedClusteringDatasetHandler`.
    4. Run clustering (e.g., HDBSCAN) through: class:`ClustererEngine`.
    5. Choose the best clustering result respecting minimum/maximum
       cluster count constraints.

    Parameters
    ----------
    embedder_path: str
        Filesystem path to the pretrained embedding model.
    embedder_input_size: int
        Expected input dimension size for the embedder.
    method: str, optional
        Clustering algorithm name (default "hdbscan").
    reduction: str, optional
        Dimensionality‑reduction technique (default "tsne").
    device: str, optional
        Compute device for the embedder (e.g., "cuda" or "cpu").
    load_model: bool, optional
        Whether to load the embedder model immediately.
    reducer_optim: str, optional
        Optimizer for the reduction algorithm (default "barnes_hut").
    reducer_sim_metric: str, optional
        Similarity metric used during reduction (default "cosine").
    clustering_params: dict | list | None, optional
        Parameters passed to the clustering engine; a single dict is
        automatically wrapped into a list.
    use_reduced_dataset: bool, optionally
        If `True`, run clustering on the reduced embeddings.
    min_cluster_count: int | None, optional
        Lower bound for acceptable number of clusters (default 15).
    opt_cluster_count: int | None, optional
        Target number of clusters used when selecting the best result
        (default 26).
    max_cluster_count: int | None, optional
        Upper bound for an acceptable number of clusters (default 38).
    """

    MIN_CLUSTER_COUNT = 15
    OPT_CLUSTER_COUNT = 26
    MAX_CLUSTER_COUNT = 38

    # Number of clusters generations for single clustering options
    NUMBER_OF_GROUPS_PER_CL_OPT = 2

    def __init__(
        self,
        embedder_path: str,
        embedder_input_size: int,
        method: str = "hdbscan",
        reduction: str = "tsne",
        device: str = "cuda",
        load_model: bool = True,
        reducer_optim: str = "barnes_hut",
        reducer_sim_metric: str = "cosine",
        clustering_params: Optional[Dict or List] = None,
        use_reduced_dataset: bool = True,
        min_cluster_count: Optional[int] = None,
        opt_cluster_count: Optional[int] = None,
        max_cluster_count: Optional[int] = None,
    ):
        """
        Create a new: class:`RdlClusterer` instance.

        All arguments are stored as instance attributes; the method also
        initializes helper handlers for datasets, embeddings, and reduction,
        and creates a class:`ClustererEngine` without any clustering
        parameters (they will be supplied later during execution).

        The `clustering_params` argument can be a single dictionary or a
        list of dictionaries; a single dictionary is wrapped into a list
        for uniform processing later in the pipeline.
        """

        self.method = method
        self.reduction = reduction
        self.reducer_optim = reducer_optim
        self.embedder_path = embedder_path

        self.min_cluster_count = (
            min_cluster_count if min_cluster_count else self.MIN_CLUSTER_COUNT
        )
        self.opt_cluster_count = (
            opt_cluster_count if opt_cluster_count else self.OPT_CLUSTER_COUNT
        )
        self.max_cluster_count = (
            max_cluster_count if max_cluster_count else self.MAX_CLUSTER_COUNT
        )

        self.use_reduced_dataset = use_reduced_dataset

        self._logger = get_logger()

        self.dataset = TextDatasetHandler(
            name=f"{method}_{reduction}", min_text_len=30, full_debug=False
        )

        self.embeddings = EmbeddingsDatasetHandler(
            embedder_path=embedder_path,
            embedder_input_size=embedder_input_size,
            device=device,
            load_model=load_model,
        )

        self.reduced_dataset = None
        if self.use_reduced_dataset:
            self.reduced_dataset = ReducedClusteringDatasetHandler(
                method=reduction, optim=reducer_optim, sim_metric=reducer_sim_metric
            )

        if clustering_params is not None:
            if type(clustering_params) is dict:
                clustering_params = [clustering_params]
        self.clustering_params = clustering_params

        self.clusterer = ClustererEngine(method=self.method, clustering_params=None)

    def load_dataset(
        self,
        file_path,
        text_column: str,
        metadata_column: Optional[str] = None,
        input_type: str = "jsonl",
        clear_dataset_if_exists: bool = True,
    ):
        """
        Load raw text data from ``file_path`` into the internal dataset.

        Parameters
        ----------
        file_path: str
            Path to the source file.
        text_column: str
            Name of the column that contains the raw text.
        metadata_column: str | None, optional
            Column name for optional per‑record metadata.
        input_type: str, optional
            Expected input format (currently only "jsonl" is supported).
        clear_dataset_if_exists: bool, optional
            If `True`, existing embeddings are cleared before loading new
            data.

        Returns
        -------
        None
            The method logs errors and returns early if loading fails.
        """
        if clear_dataset_if_exists:
            self.embeddings.clear()

        is_loaded = self.dataset.load(
            file_path=file_path,
            text_column=text_column,
            metadata_column=metadata_column,
            input_type=input_type,
            clear_dataset_if_exists=clear_dataset_if_exists,
            only_unique_texts=True,
        )

        if not is_loaded:
            self._logger.error(f"Error while loading dataset.")
            return

    def run(self, debug: bool = False, normalize_embeddings: bool = True) -> bool:
        """
        Execute the full clustering pipeline.

        The method performs the following steps:

        1. Verify that a dataset has been loaded.
        2. Vectorize the dataset using the configured embedder.
        3. Run clustering (optionally on reduced embeddings) with all
           supplied clustering parameter sets.
        4. Choose the best clustering configuration according to the
           `min_cluster_count` / `opt_cluster_count` / `max_cluster_count`
           constraints.

        Parameters
        ----------
        debug: bool, optional (default False),
            When `True` additional debug information is logged.
        normalize_embeddings: bool, optional (default True),
            When `True` embeddings will be normalized.

        Returns
        -------
        bool
            `True` if a valid clustering result was produced; otherwise `False`.
        """
        if not self.dataset.is_loaded():
            self._logger.error(f"Dataset is not loaded.")
            self._logger.error("Please call load_dataset() first.")
            return False

        # 1 - vectorize dataset
        self._logger.info(f"Vectorizing dataset with {self.embedder_path}")
        self.__vectorize_dataset(normalize_embeddings=normalize_embeddings)
        if self.embeddings.vectors is None or not len(self.embeddings.vectors):
            self._logger.error("No converted dataset found! Aborted!")
            return False

        # 2 - run clustering
        self._logger.info(f"Dataset is ready, {self.method} clustering will be run")
        self.__run_clustering(
            on_reduced=self.use_reduced_dataset,
            choose_best=True,
            number_of_cluster_groups=self.NUMBER_OF_GROUPS_PER_CL_OPT,
            debug=debug,
        )
        if (
            self.clusterer.clustering_model is None
            or self.clusterer.labels is None
            or not len(self.clusterer.labels)
        ):
            self._logger.error(
                "No clustering model found (or bad clustering)! Aborted!"
            )
            return False

        self._logger.info(f"Best clustering options found {self.clusterer.params}")

        return True

    def __vectorize_dataset(self, normalize_embeddings: bool = True):
        """
        Convert raw texts to embedding vectors.

        The method delegates to class:`EmbeddingsDatasetHandler.vectorize`,
        storing the resulting vectors in `self.embeddings.vectors`.
        """
        assert self.embeddings is not None

        self.embeddings.vectorize(
            texts=self.dataset.dataset,
            store_vector=True,
            store_text_str=False,
            show_progress=True,
            normalize_embeddings=normalize_embeddings,
        )

    def __reduce_embeddings(self):
        """
        Reduce the dimensionality of the previously computed embeddings.

        The reduction is performed by: class:`ReducedClusteringDatasetHandler`
        and the reduced vectors are stored in `self.reduced_dataset.dataset`.
        """
        assert self.reduced_dataset is not None

        self.reduced_dataset.reduce_dataset(
            embeddings=np.array(self.embeddings.vectors), store_vectors=True
        )

    def __run_clustering(
        self,
        on_reduced: bool,
        choose_best: bool,
        number_of_cluster_groups: int,
        debug: bool = False,
    ):
        """
        Run clustering for each set of clustering parameters.

        Parameters
        ----------
        on_reduced: bool
            If `True` clustering is performed on the reduced embeddings;
            otherwise the original embeddings are used.
        choose_best: bool
            When `True` all clustering runs are evaluated, and the best
            configuration is selected; otherwise the first successful run is kept.
        number_of_cluster_groups: int
            Number of times each clustering configuration is executed
            (useful for stochastic algorithms).
        debug: bool, optional
            Enables additional logging inside the reduction step.

        Returns
        -------
        None
            Results are stored in `self.clusterer.labels`
            and `self.clusterer.params`.
        """
        if number_of_cluster_groups < 1:
            number_of_cluster_groups = 1

        clusters_groups = []
        with tqdm.tqdm(
            total=len(self.clustering_params) * number_of_cluster_groups,
            desc="Running clustering",
        ) as pbar:
            for cl_opts in self.clustering_params:
                for i in range(number_of_cluster_groups):
                    # Reduce dataset if reduced dataset will be used
                    if self.use_reduced_dataset:
                        if debug:
                            self._logger.info(
                                f"Reducing embeddings with method {self.reduction}"
                                f" and {self.reducer_optim} reduction optimizer"
                            )

                        self.__reduce_embeddings()
                        if self.reduced_dataset.dataset is None or not len(
                            self.reduced_dataset.dataset
                        ):
                            self._logger.error("No reduced dataset found! Aborted!")
                            return

                    dataset_to_cluster = (
                        self.reduced_dataset.dataset
                        if on_reduced
                        else self.embeddings.vectors
                    )

                    self.clusterer.prepare_clusters(
                        dataset=dataset_to_cluster, options=cl_opts, debug=debug
                    )

                    clusters_groups.append(
                        {
                            "labels": self.clusterer.labels,
                            "options": self.clusterer.params,
                        }
                    )
                    pbar.update(1)

                if not choose_best:
                    break

        if len(clusters_groups) == 1 or not choose_best:
            if not len(clusters_groups):
                raise Exception("No clusters_groups found (choose_best=False)!")
            self.clusterer.labels = clusters_groups[0]["labels"]
            self.clusterer.params = clusters_groups[0]["options"]
            return

        self._logger.info(f"Number of clusters groups: {len(clusters_groups)}")
        self.__choose_best_clusters(clusters_groups=clusters_groups)

    def __choose_best_clusters(self, clusters_groups: list[dict]):
        """
        Select the best clustering result from `clusters_groups`.

        The selection strategy:

        * Convert each clustering group's label list to a frequency map.
        * Filter groups whose number of clusters falls within the
          `min_cluster_count` – `max_cluster_count` range.
        * Among the filtered groups, choose those whose cluster count is
          closest to `opt_cluster_count`.
        * If multiple candidates remain, prefer the one with the fewest
          outlier points (label `-1` by default).

        The chosen group's labels and parameters are stored back into
        `self.clusterer`.
        """
        self._logger.info("Choosing best clusters groups")

        groups_frequency = self.__convert_clusters_groups_to_freq(
            clusters_groups=clusters_groups
        )
        assert len(groups_frequency) == len(clusters_groups)

        best_matches_indexes = self.__best_mach_to_min_max_clusters(
            groups_frequency=groups_frequency
        )
        if not len(best_matches_indexes):
            self._logger.error("No best clusters found! Aborted!")
            for gf in groups_frequency:
                self._logger.error(gf)
            return

        nearest_idx = self.__chose_nearest_to_best_match_clusters_count(
            alls_bests_ids=best_matches_indexes,
            groups_frequency=groups_frequency,
        )
        if nearest_idx is None:
            self._logger.error("No nearest clusters found! Aborted!")
            return

        self._logger.info(f"Best matched index of clusters group {nearest_idx}")

        self.clusterer.labels = [
            int(i) for i in clusters_groups[nearest_idx]["labels"]
        ]
        self.clusterer.params = clusters_groups[nearest_idx]["options"]

    def __best_mach_to_min_max_clusters(
        self, groups_frequency: list[dict]
    ) -> list[int]:
        """
        Identify cluster groups whose size respects min/max constraints.

        Parameters
        ----------
        groups_frequency: list[dict]
            List of label‑frequency dictionaries, one per clustering run.

        Returns
        -------
        list[int]
            Indices of groups whose number of clusters lies between
            `self.min_cluster_count` and `self.max_cluster_count`.
        """
        self._logger.info("  -> choosing best match clusters count")

        alls_bests_ids = self.__choose_proper_min_max_clusters_ids(
            groups_frequency=groups_frequency
        )
        if not len(alls_bests_ids):
            self._logger.error("No proper clusters found! Aborted!")
            return []

        self._logger.info(
            f"  -> matched min/max size clusters indexes: {alls_bests_ids}"
        )
        return alls_bests_ids

    def __chose_nearest_to_best_match_clusters_count(
        self,
        alls_bests_ids: list[int],
        groups_frequency: list[dict],
    ) -> Optional[int]:
        """
        From the candidate groups, pick the one whose cluster count is
        nearest to `self.opt_cluster_count`.

        If several groups share the same distance, the method delegates
        to `__choose_with_smaller_outlier_group` to break the tie.

        Returns
        -------
        int | None
            Index of the selected group, or `None` if no suitable group is found.
        """

        self._logger.info("  -> choosing nearest to best match clusters group")

        b2size = {}
        for idx in alls_bests_ids:
            b2size[idx] = len(groups_frequency[idx])
            self._logger.info(f"    -> matched [{idx}] cluster size: {b2size[idx]}")

        nearest_diff = 99999999
        nearest_indexes = []
        for idx, g_size in b2size.items():
            diff = abs(self.opt_cluster_count - g_size)
            if diff < nearest_diff:
                nearest_diff = diff
                nearest_indexes = [idx]
            elif diff == nearest_diff:
                nearest_indexes.append(idx)

        self._logger.info(f"  -> nearest clusters difference: {nearest_diff}")
        self._logger.info(f"  -> nearest clusters indexes: {nearest_indexes}")
        if not len(nearest_indexes):
            self._logger.error("No nearest clusters found! Aborted!")
            return None

        return self.__choose_with_smaller_outlier_group(
            indexes=nearest_indexes, groups_frequency=groups_frequency
        )

    def __choose_with_smaller_outlier_group(
        self, indexes: list[int], groups_frequency: list[dict], outlier_id: int = -1
    ) -> Optional[int]:
        """
        Select the group with the smallest number of outlier points.

        Parameters
        ----------
        indexes: list[int]
            Candidate group indices.
        groups_frequency: list[dict]
            Frequency maps for all groups.
        outlier_id: int, optional
            Label used to denote outliers (default `-1`).

        Returns
        -------
        int | None
            Index of the chosen group, or `None` if `indexes` is empty.
        """
        self._logger.info("  -> choosing with smaller outlier group")

        smaller_outliers = []
        smaller_outliers_count = 99999999
        for idx in indexes:
            o_count = groups_frequency[idx].get(outlier_id, 0)
            self._logger.info(
                f"    -> {idx} with outliers {o_count} "
                f"in cluster size {len(groups_frequency[idx])}"
            )
            if o_count < smaller_outliers_count:
                smaller_outliers_count = o_count
                smaller_outliers = [idx]
            elif o_count == smaller_outliers_count:
                smaller_outliers.append(idx)

        self._logger.info(f"  -> smaller outliers count: {smaller_outliers_count}")
        self._logger.info(f"  -> smaller outliers indexes: {smaller_outliers}")

        if len(smaller_outliers) > 1:
            self._logger.info(
                f"  -> multiple smaller outliers found, choosing first one"
            )

        return smaller_outliers[0]

    def __choose_proper_min_max_clusters_ids(
        self, groups_frequency: list[dict]
    ) -> list[int]:
        """
        Return indices of groups whose cluster count lies within the
        configured minimum and maximum bounds.

        This helper is used by `__best_mach_to_min_max_clusters`.
        """
        proper_groups = []
        for idx, g in enumerate(groups_frequency):
            if self.min_cluster_count <= len(g) <= self.max_cluster_count:
                proper_groups.append(idx)
        return proper_groups

    def __convert_clusters_groups_to_freq(
        self, clusters_groups: list[dict]
    ) -> list[dict]:
        """
        Convert each clustering group's label list into a frequency map.

        The returned list contains one dictionary per group, where keys are
        integer cluster labels and values are the number
        of occurrences of that label in the group.
        """
        self._logger.info("  -> converting clusters groups to frequency of labels")

        cl_group_sizes = []
        for cl_group in clusters_groups:
            single_group_freq = {}
            for l in cl_group["labels"]:
                l = int(l)
                if l not in single_group_freq:
                    single_group_freq[l] = 0
                single_group_freq[l] += 1
            cl_group_sizes.append(single_group_freq)
        return cl_group_sizes
