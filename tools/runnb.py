"""Execute a course notebook, export figures/GIFs, and (with --inplace) save outputs back.

Usage: python runnb.py <notebook.ipynb> [--inplace]
GIFs of animations named via ec.animate(..., name=...) go to assets/<module_dir>/.
"""
import base64, os, sys
import nbformat
from nbclient import NotebookClient

def slim(nb):
    """Aligera el notebook guardado: si un gráfico Plotly tiene su PNG, se guarda sólo la PNG
    (al ejecutar en Colab se vuelve a generar la versión interactiva) y se quitan avisos de kaleido."""
    for c in nb.cells:
        outs = []
        for o in c.get("outputs", []):
            d = o.get("data", {})
            if "image/png" in d and "application/vnd.plotly.v1+json" in d:
                d.pop("application/vnd.plotly.v1+json")
                d.pop("text/html", None)
            html = d.get("text/html", "")
            if isinstance(html, list): html = "".join(html)
            if 'class="animation"' in html:          # animación jshtml: pesa MB y GitHub no la reproduce
                o["data"] = {"text/plain": "▶️ Ejecute esta celda para ver la animación con controles "
                                           "(la vista previa GIF está en la celda de texto de arriba)."}
            if o.get("output_type") == "stream" and "unclean kill browser" in o.get("text", ""):
                o["text"] = "".join(l for l in o["text"].splitlines(True) if "unclean kill browser" not in l)
                if not o["text"].strip():
                    continue
            outs.append(o)
        if c.cell_type == "code":
            c["outputs"] = outs


src = os.path.abspath(sys.argv[1]); inplace = "--inplace" in sys.argv
nb_dir = os.path.dirname(src); name = os.path.basename(src)[:-6]
repo = os.path.dirname(nb_dir)
os.environ["COURSE_ASSETS"] = os.path.join(repo, "assets", os.path.basename(nb_dir))
before = set(os.listdir(nb_dir))
nb = nbformat.read(src, as_version=4)
NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": nb_dir}}).execute()
for f in set(os.listdir(nb_dir)) - before:            # borrar archivos generados por el notebook
    p = os.path.join(nb_dir, f)
    if os.path.isfile(p) and not f.endswith(".ipynb"):  # nunca borrar notebooks (otra lección pudo crearse en paralelo)
        os.remove(p)
slim(nb)
out = os.path.join(os.environ.get("RUN_DIR", "run"), name); os.makedirs(out, exist_ok=True)
k = 0
for i, c in enumerate(nb.cells):
    for o in c.get("outputs", []):
        d = o.get("data", {})
        if "image/png" in d:
            k += 1; open(f"{out}/cell{i:02d}_{k}.png", "wb").write(base64.b64decode(d["image/png"]))
        if o.get("output_type") == "error": print("ERROR cell", i, o["ename"], o["evalue"])
nbformat.write(nb, src if inplace else f"{out}/{name}.ipynb")
print("ok", name, "images", k, "KB", os.path.getsize(src if inplace else f"{out}/{name}.ipynb") // 1024)
