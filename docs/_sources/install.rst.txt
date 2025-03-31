Installation
=============================

System requirements
-------------------------
- Python (version >= 3.11).

Before installation
--------------------------

Install `lys`(https://lys-devel.github.io/lys/index.html)

Installation from source
--------------------------------------------------------

If you want to install `lys_em` from source, follow the instructions below.

1. Update pip::

    pip install --upgrade pip

2. Clone lys_em. If you do not have git, you can download the source code from GitHub (https://github.com/a-tock/lys_em)::

    git clone git@github.com:a-tock/lys_em.git

3. Install lys by pip. If you want to install lys in development mode, add `-e` option after `pip install`::

    cd lys_em
    pip install .

4. Start lys with lys_em by the command below. Note that the current directory of the system is used as the working directory of lys::

    python -m lys -p lys_em