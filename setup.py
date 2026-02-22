"""A setup configuration for the cognitive_workflow_kit package.

This script allows the package to be installed in editable mode so that
project modules are available in the PYTHONPATH automatically.
"""

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

from setuptools import find_packages
from setuptools import setup

setup(
    name="cwk",
    version="0.1.0",
    description="A toolkit for cognitive workflow management",
    author="chanwcom",
    # find_packages() locates all directories containing an __init__.py file.
    packages=find_packages(),
    # Requirements listed here will be installed by pip when this package
    # is installed.
    install_requires=[
        # "numpy>=1.21.0",
        # "pandas>=1.3.0",
    ],
    # Useful for ensuring the code runs on appropriate Python versions.
    python_requires=">=3.10",
    include_package_data=True,
    zip_safe=False,
)
