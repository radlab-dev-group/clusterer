"""
Configuration loader for the clustering pipeline.

Parses a JSON configuration file and exposes strongly typed attributes for
clusterer settings (embedder, reduction, device, and clustering parameters),
while keeping raw subsections for potential labeler configuration.
"""

import json

from clusterer.utils import get_logger


class ClustererConfig:
    """
    Load and expose clusterer configuration from a JSON file.

    The configuration file is expected to contain at least two top-level
    sections:
    - "labeller": kept as a raw dictionary (currently not parsed).
    - "clusterer": parsed into explicit attributes used by the pipeline.

    Attributes
    ----------
    embedder_path : str | None
    embedder_input_size : int | None
    method : str | None
    reduction : str | None
    reducer_optim : str | None
    reducer_sim_metric : str | None
    device : str | None
    load_model : bool | None
    use_reduced_dataset : bool | None
    clustering_params : dict | list | None
        Parameters to pass to the clustering engine.
    labeller_config_dict : dict | None
        Raw labeller configuration subsection.
    clusterer_config_dict : dict | None
        Raw clusterer configuration subsection.
    """

    LABELLER_JSON_FIELD = "labeller"
    CLUSTERER_JSON_FIELD = "clusterer"

    def __init__(self, config_file_path: str):
        """
        Initialize the config object and optionally load from a file.

        Parameters
        ----------
        config_file_path : str
            Path to the JSON configuration file. If non-empty, the file
            is parsed immediately and attributes are populated.
        """

        self.embedder_path = None
        self.embedder_input_size = None
        self.method = None
        self.reduction = None
        self.reducer_optim = None
        self.reducer_sim_metric = None
        self.device = None
        self.load_model = None
        self.use_reduced_dataset = None
        self.clustering_params = None

        self.labeller_config_dict = None
        self.clusterer_config_dict = None

        self._logger = get_logger()

        self.config_file_path = config_file_path
        if len(config_file_path):
            self.__load()

    def __load(self):
        """
        Read the JSON file and dispatch parsing to section handlers.

        Populates:
        - labeller_config_dict
        - clusterer_config_dict

        Then calls the dedicated loaders for each section.
        """
        self._logger.info(f"Loading config from {self.config_file_path}")
        with open(self.config_file_path, "r") as f:
            config = json.load(f)
            self.labeller_config_dict = config[self.LABELLER_JSON_FIELD]
            self.clusterer_config_dict = config[self.CLUSTERER_JSON_FIELD]
        self.__load_labeller_section()
        self.__load_clustering_section()

    def __load_labeller_section(self):
        """Parse the 'labeller' section of the config.

        Currently, a placeholder. Keeps the raw dictionary in
        labeller_config_dict for future use.
        """
        pass

    def __load_clustering_section(self):
        """
        Parse the 'clusterer' section and populate attributes.

        Expects keys such as
        - embedder_path, embedder_input_size
        - method, reduction, reducer_optim, reducer_sim_metric
        - device, load_model, use_reduced_dataset
        - clustering_params

        Raises
        ------
        AssertionError
            If the clusterer_config_dict is missing.
        """
        assert self.clusterer_config_dict is not None

        self._logger.info(f"Loading clusterer {self.CLUSTERER_JSON_FIELD} section")

        self.embedder_path = self.clusterer_config_dict.get("embedder_path")
        self.embedder_input_size = self.clusterer_config_dict.get(
            "embedder_input_size"
        )
        self.method = self.clusterer_config_dict.get("method")
        self.reduction = self.clusterer_config_dict.get("reduction")
        self.reducer_optim = self.clusterer_config_dict.get("reducer_optim")
        self.reducer_sim_metric = self.clusterer_config_dict.get(
            "reducer_sim_metric"
        )
        self.device = self.clusterer_config_dict.get("device")
        self.load_model = self.clusterer_config_dict.get("load_model")
        self.use_reduced_dataset = self.clusterer_config_dict.get(
            "use_reduced_dataset"
        )
        self.clustering_params = self.clusterer_config_dict.get("clustering_params")
