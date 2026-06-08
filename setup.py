"""Package configuration for mapMyVault."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip()]

setup(
    name="mapMyVault",
    version="2.0.0",
    description="Fully local, resumable repository intelligence with an Obsidian vault and MCP tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sukantsondhi/mapMyVault",
    packages=find_packages(exclude=("tests", "tests.*")),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Office/Business",
        "Topic :: Utilities",
        "Development Status :: 4 - Beta",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "mapmyvault=src.cli:main",
        ],
    },
)
