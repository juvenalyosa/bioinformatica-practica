"""Helpers to build course notebooks with nbformat."""
import textwrap
import nbformat as nbf

REPO = "juvenalyosa/bioinformatica-practica"


def colab_badge(path):
    return (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/{REPO}/blob/main/{path})")


SETUP = '''\
# ⚙️ Configuración del entorno (ejecute esta celda primero)
import os, sys, urllib.request

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Descarga el estilo gráfico del curso (en Colab) o lo toma del repositorio (local)
STYLE_URL = "https://raw.githubusercontent.com/juvenalyosa/bioinformatica-practica/main/utils/estilo_curso.py"
if os.path.exists("../utils/estilo_curso.py"):
    sys.path.insert(0, "../utils")
elif not os.path.exists("estilo_curso.py"):
    urllib.request.urlretrieve(STYLE_URL, "estilo_curso.py")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import estilo_curso as ec

ec.set_style()
print("Entorno listo ✔  |  Colab:", IN_COLAB)
'''


class NB:
    def __init__(self):
        self.nb = nbf.v4.new_notebook()
        self.nb.metadata = {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        }

    def md(self, s):
        self.nb.cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip("\n")))

    def code(self, s):
        self.nb.cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip("\n")))

    def save(self, path):
        nbf.write(self.nb, path)
