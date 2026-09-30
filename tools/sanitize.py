"""Limpia de las salidas guardadas las rutas del equipo donde se construyó el curso.

Los avisos (stderr) que citan archivos locales se eliminan; en stdout, las rutas absolutas locales se reducen al
nombre del archivo o herramienta. Uso: python sanitize.py nb1.ipynb [nb2.ipynb ...]  (o importar sanitize_nb).
"""
import re, sys
import nbformat

LOCAL = r"(?:/private/tmp|/private/var/folders|/var/folders|/Users/[^/\s]+|/opt/anaconda3|/tmp/claude-\d+)"
PATH_RE = re.compile(LOCAL + r"[^\s'\"),:;]*")
WARN_RE = re.compile(LOCAL + r"\S*:\d+: \w*(Warning|Error)")


def _clean_text(text, stream):
    out, skip_next = [], False
    for line in text.splitlines(True):
        if skip_next:                      # la línea de código que Python repite tras un aviso
            skip_next = False
            if line.startswith("  "):
                continue
        if stream == "stderr" and WARN_RE.search(line):
            skip_next = True
            continue
        out.append(PATH_RE.sub(lambda m: m.group(0).rstrip("/").split("/")[-1], line))
    return "".join(out)


def sanitize_nb(nb):
    n = 0
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        keep = []
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                new = _clean_text(o["text"], o.get("name"))
                n += new != o["text"]
                if not new.strip():
                    continue
                o["text"] = new
            elif "text/plain" in o.get("data", {}):
                t = o["data"]["text/plain"]
                new = PATH_RE.sub(lambda m: m.group(0).rstrip("/").split("/")[-1], t)
                n += new != t
                o["data"]["text/plain"] = new
            keep.append(o)
        c["outputs"] = keep
    return n


if __name__ == "__main__":
    for p in sys.argv[1:]:
        nb = nbformat.read(p, as_version=4)
        k = sanitize_nb(nb)
        if k:
            nbformat.write(nb, p)
        print(f"{k:4d} salidas limpiadas · {p}")
