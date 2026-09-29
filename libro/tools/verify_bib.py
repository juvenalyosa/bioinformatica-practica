#!/usr/bin/env python3
"""Verifica que cada referencia de bib/capNN.bib EXISTA y esté bien citada.

Uso:  python3 tools/verify_bib.py 03            (un capítulo)
      python3 tools/verify_bib.py all           (todos)

Reglas (cualquier fallo => código de salida 1):
  * Entradas con `doi`: Crossref (o DataCite si Crossref da 404) debe
    devolver 200 y coincidir título (similitud >= 0.80), año (±1, por
    diferencias online/impreso) y apellido del primer autor.
  * @book / @incollection sin DOI: `isbn` verificado en OpenLibrary o
    Google Books, con título coincidente.
  * @online / @software / @manual sin DOI ni ISBN: la `url` debe responder.
  * Una entrada sin doi, isbn ni url => FALLA.
Escribe build/verify/capNN.json con el detalle.
"""
import difflib, json, re, sys, time, unicodedata, urllib.parse
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
MAILTO = "juvenal.yosa@gmail.com"
S = requests.Session()
S.headers["User-Agent"] = f"bioinfo-libro-verify/1.0 (mailto:{MAILTO})"
_tag = re.sub(r"\W", "", sys.argv[1]) if len(sys.argv) > 1 else "all"
CACHE_P = ROOT / "build" / "verify" / f"_cache_{_tag}.json"  # uno por proceso (agentes en paralelo)
CACHE_P.parent.mkdir(parents=True, exist_ok=True)
CACHE = json.loads(CACHE_P.read_text()) if CACHE_P.exists() else {}


# ------------------------------------------------------------------ parser
def parse_bib(text):
    entries, i = [], 0
    while True:
        m = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,").search(text, i)
        if not m:
            break
        typ, key, j = m.group(1).lower(), m.group(2), m.end()
        depth, k = 1, j
        while k < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        body, i = text[j:k - 1], k
        if typ in ("comment", "string", "preamble"):
            continue
        fields, p = {}, 0
        fre = re.compile(r"\s*([\w-]+)\s*=\s*")
        while True:
            fm = fre.match(body, p)
            if not fm:
                break
            name, p = fm.group(1).lower(), fm.end()
            if p < len(body) and body[p] == "{":
                d, q = 1, p + 1
                while q < len(body) and d:
                    d += {"{": 1, "}": -1}.get(body[q], 0)
                    q += 1
                val, p = body[p + 1:q - 1], q
            elif p < len(body) and body[p] == '"':
                q = body.index('"', p + 1)
                val, p = body[p + 1:q], q + 1
            else:
                q = re.compile(r"[,\n]").search(body, p)
                q = q.start() if q else len(body)
                val, p = body[p:q].strip(), q
            fields[name] = val.strip()
            c = re.compile(r"\s*,?").match(body, p)
            p = c.end()
        entries.append({"type": typ, "key": key, **fields})
    return entries


# ------------------------------------------------------------ normalizers
def norm(s):
    s = re.sub(r"\\[a-zA-Z]+\s*", " ", s or "")
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"<[^>]+>", " ", s)          # etiquetas <i> de Crossref
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def sim(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    r = difflib.SequenceMatcher(None, a, b).ratio()
    # Títulos con subtítulo: aceptar si uno contiene al otro
    if len(a) > 15 and len(b) > 15 and (a in b or b in a):
        r = max(r, 0.95)
    return r


def first_surname(author_field):
    if not author_field:
        return ""
    first = re.split(r"\s+and\s+", author_field)[0].strip()
    if first.startswith("{") and first.endswith("}"):  # autor corporativo
        return norm(first)
    last = first.split(",")[0] if "," in first else first.split()[-1]
    return norm(last)


def get(url, **kw):
    if url in CACHE:
        return CACHE[url]
    err = "sin respuesta"
    for attempt in range(4):
        try:
            r = S.get(url, timeout=25, **kw)
            if r.status_code in (429, 503):
                err = f"HTTP {r.status_code}"
                time.sleep(2 + 3 * attempt)
                continue
            out = {"status": r.status_code,
                   "json": r.json() if "json" in r.headers.get("content-type", "") else None}
            if r.status_code in (200, 404):
                CACHE[url] = out
            return out
        except Exception as e:  # red inestable
            err = str(e)
            time.sleep(1 + attempt)
    return {"status": -1, "error": err}


# -------------------------------------------------------------- checkers
def check_doi(e):
    doi = e["doi"].strip().removeprefix("https://doi.org/").removeprefix("http://dx.doi.org/")
    r = get(f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='/')}?mailto={MAILTO}")
    src = "crossref"
    if r["status"] == 200:
        m = r["json"]["message"]
        title = " ".join(m.get("title") or [""])
        sub = " ".join(m.get("subtitle") or [])
        ys = [m.get(k, {}).get("date-parts", [[None]])[0][0]
              for k in ("published-print", "published-online", "issued", "published")]
        auth = m.get("author") or m.get("editor") or []
        surnames = [norm(a.get("family") or a.get("name", "")) for a in auth]
    else:
        r = get(f"https://api.datacite.org/dois/{urllib.parse.quote(doi, safe='/')}")
        src = "datacite"
        if r["status"] != 200:
            return False, f"DOI {doi} no existe (Crossref y DataCite)"
        a = r["json"]["data"]["attributes"]
        title, sub = (a.get("titles") or [{}])[0].get("title", ""), ""
        ys = [a.get("publicationYear")]
        surnames = [norm(c.get("familyName") or c.get("name", "")) for c in a.get("creators", [])]
    s = max(sim(e.get("title", ""), title), sim(e.get("title", ""), f"{title} {sub}"))
    problems = []
    if s < 0.80:
        problems.append(f"título no coincide ({s:.2f}): bib='{e.get('title','')[:70]}' vs {src}='{title[:70]}'")
    year = re.search(r"\d{4}", e.get("date", "") or e.get("year", ""))
    ys = [int(y) for y in ys if y]
    if not year or not ys or all(abs(int(year.group()) - y) > 1 for y in ys):
        problems.append(f"año no coincide: bib={year.group() if year else None} vs {sorted(set(ys))}")
    fa = first_surname(e.get("author") or e.get("editor", ""))
    if surnames and fa and not any(fa == x or fa in x or (x and x in fa) for x in surnames[:1]):
        # tolera orden distinto solo si aparece entre los 3 primeros
        if not any(fa == x or fa in x for x in surnames[:3]):
            problems.append(f"primer autor no coincide: bib='{fa}' vs {src}={surnames[:3]}")
    return (not problems), "; ".join(problems) or f"ok {src} (título {s:.2f})"


def check_isbn(e):
    isbn = re.sub(r"[^0-9Xx]", "", e["isbn"])
    r = get(f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&jscmd=data&format=json")
    titles = []
    if r["status"] == 200 and r["json"]:
        d = r["json"].get(f"ISBN:{isbn}", {})
        titles.append(d.get("title", "") + " " + d.get("subtitle", ""))
    r3 = get(f"https://openlibrary.org/search.json?isbn={isbn}&fields=title,subtitle&limit=3")
    if r3["status"] == 200 and r3["json"]:
        titles += [d.get("title", "") + " " + d.get("subtitle", "") for d in r3["json"].get("docs", [])]
    r2 = get(f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}")
    if r2["status"] == 200 and r2["json"]:
        for it in r2["json"].get("items", []) or []:
            v = it["volumeInfo"]
            titles.append(v.get("title", "") + " " + v.get("subtitle", ""))
    titles = [t.strip() for t in titles if t.strip()]
    if not titles:
        return False, f"ISBN {isbn} no encontrado (OpenLibrary/Google Books)"
    bt = e.get("title", "") if e["type"] == "book" else e.get("booktitle", e.get("title", ""))
    s = max(max(sim(bt, t), sim(bt, t.split(":")[0])) for t in titles)
    if s < 0.75:
        return False, f"ISBN {isbn}: título no coincide ({s:.2f}) '{bt[:60]}' vs '{titles[0][:60]}'"
    return True, f"ok isbn (título {s:.2f})"


def check_url(e):
    url = e["url"]
    if url in CACHE and CACHE[url].get("status") == 200:
        return True, "ok url (cache)"
    try:
        r = S.get(url, timeout=25, allow_redirects=True)
        code = r.status_code
    except Exception as ex:
        return False, f"url inaccesible: {ex}"
    if code in (200, 403):  # 403: sitios que bloquean bots pero existen
        CACHE[url] = {"status": 200}
        return True, f"ok url ({code})"
    return False, f"url responde {code}"


def check(e):
    if e.get("doi"):
        return check_doi(e)
    if e.get("isbn"):
        return check_isbn(e)
    if e.get("url") and e["type"] in ("online", "software", "manual", "misc", "dataset", "report"):
        return check_url(e)
    return False, "sin doi/isbn/url verificable"


def run(n):
    bib = ROOT / "bib" / f"cap{n}.bib"
    if not bib.exists():
        print(f"cap{n}: no existe {bib}")
        return False
    entries = parse_bib(bib.read_text(encoding="utf-8"))
    report, bad = [], 0
    for e in entries:
        ok, msg = check(e)
        bad += not ok
        report.append({"key": e["key"], "ok": ok, "msg": msg})
        print(f"  {'✔' if ok else '✘'} {e['key']:<28} {msg}")
        time.sleep(0.05)
    out = ROOT / "build" / "verify" / f"cap{n}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    CACHE_P.write_text(json.dumps(CACHE))
    print(f"cap{n}: {len(entries) - bad}/{len(entries)} verificadas" + ("" if not bad else f"  — {bad} FALLAS"))
    return bad == 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    caps = sorted(p.stem[3:] for p in (ROOT / "bib").glob("cap*.bib")) if arg == "all" \
        else [f"{int(arg):02d}"]
    ok = all([run(c) for c in caps])
    sys.exit(0 if ok else 1)
