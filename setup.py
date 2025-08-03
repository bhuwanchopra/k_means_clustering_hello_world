from setuptools import setup, find_packages

setup(
    name="k_means_clustering_hello_world",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pytest>=7.0.0",
    ],
)
