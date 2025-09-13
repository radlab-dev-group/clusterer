import json

from clusterer.clustering.clusterer import RdlClusterer
from clusterer.clustering.config import ClustererConfig


def main(argv=None):
    dataset_jsonl = "/tmp/tmpbhfrxolz"

    config = ClustererConfig(config_file_path="configs/clusterer-config-small.json")

    c = RdlClusterer(
        embedder_path=config.embedder_path,
        embedder_input_size=config.embedder_input_size,
        method=config.method,
        reduction=config.reduction,
        reducer_optim=config.reducer_optim,
        reducer_sim_metric=config.reducer_sim_metric,
        device=config.device,
        load_model=True,
        use_reduced_dataset=config.use_reduced_dataset,
        clustering_params=config.clustering_params,
    )

    c.load_dataset(
        file_path=dataset_jsonl,
        text_column="text",
        metadata_column="metadata",
        input_type="jsonl",
        clear_dataset_if_exists=True,
    )

    if not c.run(debug=False, normalize_embeddings=True):
        return


if __name__ == "__main__":
    main()
