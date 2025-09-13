from setuptools import setup

setup(
    name="clusterer",
    version="0.0.1",
    description="Handling clustering process",
    author="RadLab Team",
    author_email="pawel@radlab.dev",
    packages=[
        "clusterer",
        "clusterer.clustering",
        "clusterer.dataset",
        "clusterer.labels",
    ],
    install_requires=["hdbscan"],
)
