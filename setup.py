from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="open-unlearning",
    version="0.1.0",
    author="Vineeth Dorna, Anmol Reddy Mekala",
    author_email="vineethdornal@gmail.com, m.anmolreddy@gmail.com",
    description="A library for machine unlearning in LLMs.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/locuslab/open-unlearning",
    packages=find_packages(),
    install_requires=requirements,  # Uses requirements.txt
    extras_require={
        "flash-attn": ["flash-attn==2.6.3"],  # Optional installation
        "dev": ["pre-commit==4.0.1"],  # Install using `pip install .[dev]`
    },
    python_requires=">=3.11",
)
