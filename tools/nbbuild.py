"""Helpers to build course notebooks with nbformat.

Autor: Juvenal Yosa, PhD (juvenal.yosa@gmail.com) · Copiloto: Claude (Anthropic)
"""
import textwrap
import nbformat as nbf

REPO = "juvenalyosa/bioinformatica-practica"
RAW = f"https://raw.githubusercontent.com/{REPO}/main"
AUTHOR = "Juvenal Yosa, PhD"
EMAIL = "juvenal.yosa@gmail.com"
CLAUDE_BADGE = ("[![Copiloto: Claude](https://img.shields.io/badge/Copiloto-Claude-D97757?logo=claude&logoColor=white)]"
                "(https://claude.ai)")
LICENSE_BADGE = f"[![Licencia: MIT](https://img.shields.io/badge/Licencia-MIT-2a78d6.svg)](https://github.com/{REPO}/blob/main/LICENSE)"


def colab_badge(path):
    return (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/{REPO}/blob/main/{path})")


def header(path, title, module, duration, level, prereq):
    return f"""# {title}

{colab_badge(path)} {CLAUDE_BADGE} {LICENSE_BADGE}

**{module}** · Bioinformática Práctica (Maestría)

👤 **{AUTHOR}** · ✉️ [{EMAIL}](mailto:{EMAIL}) · 🤖 Copiloto: **Claude** (Anthropic)

| ⏱️ Duración | 📶 Nivel | 🧰 Requisitos |
|---|---|---|
| {duration} | {level} | {prereq} |

---
"""


FOOTER = f"""---

<sub>**Bioinformática Práctica** · © 2026 {AUTHOR} · [{EMAIL}](mailto:{EMAIL}) · Código bajo licencia MIT ·
Desarrollado con **Claude** (Anthropic) como copiloto.</sub>

{CLAUDE_BADGE}
"""


def gif(module_dir, name, caption):
    """Markdown que muestra la vista previa GIF de una animación (guardada en assets/)."""
    return f"![{caption}]({RAW}/assets/{module_dir}/{name}.gif)\n\n*{caption}*"


SETUP = '''\
# ⚙️ Configuración del entorno (ejecute esta celda primero)
# Bioinformática Práctica · Juvenal Yosa, PhD · Copiloto: Claude
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
            "authors": [{"name": AUTHOR, "email": EMAIL}],
        }

    def md(self, s):
        self.nb.cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip("\n")))

    def code(self, s):
        self.nb.cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip("\n")))

    def save(self, path):
        self.md(FOOTER)
        nbf.write(self.nb, path)
