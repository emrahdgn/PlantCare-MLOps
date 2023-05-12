from pathlib import Path

from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]

docs_packages = ["mkdocs==1.4.3", "mkdocstrings==0.21.2", "mkdocstrings-python==1.0.0"]

style_packages = ["black==23.3.0", "flake8==6.0.0", "isort==5.12.0"]

test_packages = ["pytest==7.3.1", "pytest-cov==4.0.0", "great-expectations==0.16.5"]

# Define our package
setup(
    name="plantcare",
    version=0.1,
    description="Identify Plant Diseases",
    author="Emrah Dogan",
    author_email="doganemrah7@gmail.com",
    url="https://github.com/emrahdgn/PlantCare-MLOps",
    python_requires=">=3.10",
    packages=find_namespace_packages(),
    install_requires=required_packages,
    extras_require={
        "dev": docs_packages + style_packages + test_packages + ["pre-commit==3.3.1"],
        "docs": docs_packages,
        "test": test_packages,
    },
)
