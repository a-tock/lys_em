from setuptools import setup, find_packages
import sys
from pathlib import Path
sys.path.append('./lys_em')
sys.path.append('./test')


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="lys_em",
    packages=find_packages(exclude=("test*",)),
    version="0.1.0",
    description="Python code for simulation and analysis of electron microscope images.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Hidenori Tsuji",
    author_email="h.tsuji.ni.mar91+git@gmail.com",
    install_requires=open('requirements.txt').read().splitlines(),
)
