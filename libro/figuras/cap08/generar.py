"""Genera datos y cifras del capítulo 8 (ensamblaje y anotación).

Ejecutar desde libro/:  python3 figuras/cap08/generar.py
Escribe .dat en figuras/cap08/ y guarda en cifras.txt todas las cifras que
se citan en el texto (única fuente de verdad).  Descarga el genoma de
E. coli K-12 MG1655 (NC_000913.3) a un directorio temporal si no existe.
"""
from pathlib import Path
from collections import Counter, defaultdict
import math
import tempfile
import urllib.request
import numpy as np
from scipy import stats, optimize

OUT = Path(__file__).parent
rng = np.random.default_rng(8)
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def w(name, text):
    (OUT / name).write_text(text + "\n")


COMP = str.maketrans("ACGT", "TGCA")


def revcomp(s):
    return s.translate(COMP)[::-1]


# =====================================================================
# 8.1  Espectro de k-mers de un genoma diploide simulado
# =====================================================================
G, K, L = 1_000_000, 21, 150
HET, ERR, COV_HAP = 0.01, 0.005, 20          # cobertura por haplotipo
genA = rng.integers(0, 4, G).astype(np.uint8)
# duplicaciones segmentarias: 15 segmentos de 2 kb copiados en otro lugar
ndup, ldup = 15, 2000
src = rng.choice(G - ldup, ndup, replace=False)
dst = rng.choice(G - ldup, ndup, replace=False)
for s, d in zip(src, dst):
    genA[d:d + ldup] = genA[s:s + ldup]
genB = genA.copy()
snp = rng.random(G) < HET
genB[snp] = (genB[snp] + rng.integers(1, 4, snp.sum())) % 4
say(f"[esp] G={G} k={K} L={L} het={HET} err={ERR} cob/hap={COV_HAP}"
    f" SNPs={snp.sum()} duplicado={ndup*ldup} pb")


def reads_de(gen, cov):
    n = int(cov * len(gen) / L)
    st = rng.integers(0, len(gen) - L + 1, n)
    R = gen[st[:, None] + np.arange(L)[None, :]]
    rc = rng.random(n) < 0.5                      # hebra aleatoria
    R[rc] = (3 - R[rc])[:, ::-1]
    e = rng.random(R.shape) < ERR
    R[e] = (R[e] + rng.integers(1, 4, e.sum())) % 4
    return R


def kmers_canon(R, k):
    """Códigos enteros de k-mers canónicos (2 bits por base)."""
    n, Lr = R.shape
    m = Lr - k + 1
    f = np.zeros((n, m), dtype=np.int64)
    r = np.zeros((n, m), dtype=np.int64)
    for j in range(k):
        f = (f << 2) | R[:, j:j + m].astype(np.int64)
        r = r | ((3 - R[:, j:j + m].astype(np.int64)) << (2 * j))
    return np.minimum(f, r).ravel()


codes = []
for gen in (genA, genB):
    R = reads_de(gen, COV_HAP)
    for i in range(0, len(R), 20000):
        codes.append(kmers_canon(R[i:i + 20000], K))
codes = np.concatenate(codes)
_, cnt = np.unique(codes, return_counts=True)
hist = np.bincount(cnt, minlength=122)
M = 100
h = hist[1:M + 1].astype(float)
mm = np.arange(1, M + 1)
say(f"[esp] k-mers totales={codes.size:,}  distintos={cnt.size:,}"
    f"  con multiplicidad 1={hist[1]:,}")
lam_teo = COV_HAP * (L - K + 1) / L
say(f"[esp] lambda teórico (por haplotipo) = {lam_teo:.2f};"
    f" pico homocigoto esperado = {2*lam_teo:.2f}")

# valle entre pico de error y pico heterocigoto
valle = int(np.argmin(h[2:12])) + 3
say(f"[esp] valle en m={valle}")


def nb(m, mu, r):
    p = r / (r + mu)
    return stats.nbinom.pmf(m, r, p)


def modelo(par, m):
    lam, a1, a2, a3, a4, r = par
    return (a1 * nb(m, lam, r) + a2 * nb(m, 2 * lam, r)
            + a3 * nb(m, 3 * lam, r) + a4 * nb(m, 4 * lam, r))


sel = mm >= valle
par0 = [lam_teo * 0.9, h[sel].max() * 10, h[sel].max() * 30, 1e3, 1e3, 20]
res = optimize.least_squares(
    lambda p: (modelo(p, mm[sel]) - h[sel]) / np.sqrt(h[sel] + 10), par0,
    bounds=([5, 0, 0, 0, 0, 1], [40, 1e8, 1e8, 1e8, 1e8, 1e4]))
lam, a1, a2, a3, a4, rr = res.x
say(f"[esp] ajuste: lambda={lam:.2f} Nhet={a1:,.0f} Nhom={a2:,.0f}"
    f" N3={a3:,.0f} N4={a4:,.0f} r={rr:.1f}")
alfa = a1 / (a1 + 2 * a2)
het_est = 1 - (1 - alfa) ** (1 / K)
say(f"[esp] alfa (fracción de posiciones con k-mer heterocigoto)={alfa:.4f}"
    f"  teórico={1-(1-HET)**K:.4f}  -> h estimada={het_est:.4%}")
tot_sin_err = (mm[sel] * h[sel]).sum() + sum(
    m * hist[m] for m in range(M + 1, len(hist)))
G_est = tot_sin_err / (2 * lam)
say(f"[esp] k-mers (m>={valle}) = {tot_sin_err:,.0f};  G estimado ="
    f" {G_est:,.0f}  (real {G:,})")
err_kmers = (mm[~sel] * h[~sel]).sum()
say(f"[esp] instancias de k-mers de error (m<{valle}) = {err_kmers:,.0f}"
    f"  ({err_kmers/codes.size:.2%});  distintos = {h[~sel].sum():,.0f}")
say(f"[esp] fracción de k-mers sin error esperada (1-e)^k ="
    f" {(1-ERR)**K:.4f}")
fit = modelo(res.x, mm)
lines = ["m obs het hom dup fit"]
for i, m in enumerate(mm):
    lines.append(f"{m} {max(h[i],0.5):.0f} {max(a1*nb(m,lam,rr),1e-3):.2f}"
                 f" {max(a2*nb(m,2*lam,rr),1e-3):.2f}"
                 f" {max(a3*nb(m,3*lam,rr)+a4*nb(m,4*lam,rr),1e-3):.2f}"
                 f" {max(fit[i],1e-3):.2f}")
w("espectro.dat", "\n".join(lines))
w("espectro_param.tex",
  "\\def\\capochovalle{%d}\\def\\capocholam{%.1f}\\def\\capochohom{%.1f}"
  % (valle, lam, 2 * lam))


# =====================================================================
# 8.1  Elección de k
# =====================================================================
lines = ["k rep5M rep3G ok1 ok01 cov150"]
for k in range(9, 64):
    r5 = 1 - math.exp(-2 * 5e6 / 4 ** k)
    r3 = 1 - math.exp(-2 * 3.1e9 / 4 ** k)
    lines.append(f"{k} {r5:.6f} {r3:.6f} {(1-0.01)**k:.4f} {(1-0.001)**k:.4f}"
                 f" {(150-k+1)/150:.4f}")
w("kelec.dat", "\n".join(lines))
for k in (11, 15, 17, 21, 31):
    say(f"[k] k={k}: P(rep azar) bacteria={1-math.exp(-2*5e6/4**k):.3g}"
        f"  humano={1-math.exp(-2*3.1e9/4**k):.3g}  (1-0.01)^k={0.99**k:.3f}")
say(f"[k] log4(5e6)={math.log(5e6,4):.2f} log4(3.1e9)={math.log(3.1e9,4):.2f}")

# =====================================================================
# 8.2  Grafo de De Bruijn de ejemplo y camino euleriano
# =====================================================================
S = "CAGGTTGCAGAAGGA"


def debruijn(seqs, k):
    g = defaultdict(list)
    for s in seqs:
        for i in range(len(s) - k + 1):
            g[s[i:i + k - 1]].append(s[i + 1:i + k])
    return g


def hierholzer(g, start):
    g = {u: list(v) for u, v in g.items()}
    pila, camino = [start], []
    while pila:
        u = pila[-1]
        if g.get(u):
            pila.append(g[u].pop())
        else:
            camino.append(pila.pop())
    return camino[::-1]


def todos_eulerianos(edges, start):
    out = set()

    def rec(u, usados, cam):
        if len(usados) == len(edges):
            out.add(cam)
            return
        for i, (a, b) in enumerate(edges):
            if i not in usados and a == u:
                rec(b, usados | {i}, cam + b[-1])
    rec(start, frozenset(), start)
    return out


for k in (4, 5):
    g = debruijn([S], k)
    edges = [(u, v) for u in g for v in g[u]]
    nodes = set(g) | {v for u in g for v in g[u]}
    p = hierholzer(g, S[:k - 1])
    sols = todos_eulerianos(edges, S[:k - 1])
    say(f"[dbg] k={k}: nodos={len(nodes)} aristas={len(edges)}"
        f" soluciones={sorted(sols)}  hierholzer={p[0]+''.join(x[-1] for x in p[1:])}")
lecturas = ["CAGGTTGC", "GTTGCAGA", "GCAGAAGG", "AGAAGGA"]
say("[dbg] lecturas ejemplo:", lecturas,
    "k-mers(4) de la 1a:", [lecturas[0][i:i+4] for i in range(5)])
# grados
g4 = debruijn([S], 4)
indeg = Counter(v for u in g4 for v in g4[u])
for u in sorted(set(g4) | set(indeg)):
    say(f"[dbg]   {u}: in={indeg[u]} out={len(g4.get(u, []))}")
# errores: k-mers espurios
for cov, e in ((30, 0.001), (30, 0.01)):
    NL = cov * 3.1e9
    say(f"[dbg] humano {cov}x e={e}: errores={NL*e:.3g};"
        f" k-mers espurios (k=31) ~ {NL*e*31:.3g}")

# =====================================================================
# 8.3  Métricas de contigüidad: N50, NG50, curva Nx
# =====================================================================
def nx_curve(lens, total=None):
    lens = np.sort(np.asarray(lens))[::-1]
    total = total or lens.sum()
    cum = np.cumsum(lens)
    xs = np.arange(0, 101)
    out = []
    for x in xs:
        idx = np.searchsorted(cum, x / 100 * total - 1e-9)
        out.append(lens[idx] if idx < len(lens) else 0)
    return xs, np.array(out)


def n50(lens, total=None, x=50):
    lens = sorted(lens, reverse=True)
    total = total or sum(lens)
    c = 0
    for i, l in enumerate(lens):
        c += l
        if c >= x / 100 * total:
            return l, i + 1
    return 0, None


ej = [320, 250, 180, 120, 90, 60, 40, 25, 10, 5]
say(f"[n50] ejemplo {ej} suma={sum(ej)}  N50,L50={n50(ej)}"
    f"  NG50,LG50 (G=1300)={n50(ej, 1300)}  N90={n50(ej, None, 90)}"
    f"  auN={sum(l*l for l in ej)/sum(ej):.1f}")
# ensamblaje con contigs artificialmente unidos (inflado)
ej2 = [320 + 250] + ej[2:]
say(f"[n50] si se unen (por error) los dos mayores: N50,L50={n50(ej2)}")

Gref = 5_000_000


def romper(G, nbreak, perdida, minlen):
    cortes = np.sort(rng.choice(G, nbreak, replace=False))
    lens = np.diff(np.r_[0, cortes, G])
    lens = lens[lens >= minlen]
    lens = (lens * (1 - perdida)).astype(int)
    return lens


A = romper(Gref, 220, 0.03, 500)
B = romper(Gref, 9, 0.002, 500)
lines = ["x A B AG BG"]
xa, na = nx_curve(A)
_, nb_ = nx_curve(B)
_, nag = nx_curve(A, Gref)
_, nbg = nx_curve(B, Gref)
for i in range(101):
    lines.append(f"{xa[i]} {na[i]/1e3:.2f} {nb_[i]/1e3:.2f} {nag[i]/1e3:.2f}"
                 f" {nbg[i]/1e3:.2f}")
w("nx.dat", "\n".join(lines))
for nom, lens in (("A", A), ("B", B)):
    say(f"[nx] {nom}: contigs={len(lens)} total={lens.sum():,}"
        f" N50={n50(lens)} NG50={n50(lens, Gref)}"
        f" auN={(lens.astype(float)**2).sum()/lens.sum():,.0f}")

# Merqury QV
for kasm, kfal in ((5_000_000, 250), (5_000_000, 5000)):
    pbien = (1 - kfal / kasm) ** (1 / K)
    E = 1 - pbien
    say(f"[qv] K_asm={kasm:,} solo_asm={kfal}: error/base={E:.3e}"
        f"  QV={-10*math.log10(E):.1f}")

# =====================================================================
# 8.4  ORFs en E. coli K-12 frente a un genoma barajado
# =====================================================================
cache = Path(tempfile.gettempdir()) / "cap08_NC_000913.3.fasta"
if not cache.exists():
    url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
           "db=nuccore&id=NC_000913.3&rettype=fasta&retmode=text")
    cache.write_bytes(urllib.request.urlopen(url, timeout=120).read())
eco = "".join(l.strip() for l in cache.read_text().splitlines()[1:]).upper()
say(f"[orf] E. coli K-12 MG1655: {len(eco):,} pb  GC={(eco.count('G')+eco.count('C'))/len(eco):.4f}")
STOP = {"TAA", "TAG", "TGA"}


def orfs(seq, minc=30):
    """ORFs ATG..stop más largos por codón de parada, 6 marcos."""
    res = []
    for hebra in (seq, revcomp(seq)):
        for f in range(3):
            ini = None
            for i in range(f, len(hebra) - 2, 3):
                c = hebra[i:i + 3]
                if c in STOP:
                    if ini is not None and (i - ini) // 3 >= minc:
                        res.append(hebra[ini:i])
                    ini = None
                elif ini is None and c == "ATG":
                    ini = i
    return res


arr = np.frombuffer(eco.encode(), dtype=np.uint8).copy()
rng.shuffle(arr)
bar = arr.tobytes().decode()
O_real, O_bar = orfs(eco), orfs(bar)
lr = np.array([len(o) // 3 for o in O_real])
lb = np.array([len(o) // 3 for o in O_bar])
freq = {b: eco.count(b) / len(eco) for b in "ACGT"}
pstop = sum(freq[a] * freq[b] * freq[c] for a, b, c in STOP)
say(f"[orf] P(stop|codón al azar)={pstop:.4f} (1/{1/pstop:.1f});"
    f"  P(ORF>=100 codones)={(1-pstop)**100:.4f}  >=300: {(1-pstop)**300:.2e}")
for t in (100, 200, 300, 500):
    say(f"[orf] ORFs >= {t} codones: real={np.sum(lr>=t):,}  barajado={np.sum(lb>=t):,}")
say(f"[orf] ORFs >=30 codones: real={len(lr):,} barajado={len(lb):,}"
    f"  máx barajado={lb.max()}  máx real={lr.max()}")
bins = np.arange(30, 1030, 20)
hr, _ = np.histogram(lr, bins)
hb, _ = np.histogram(lb, bins)
# teoría: nº de ORFs que empiezan en ATG con longitud n en genoma aleatorio
lines = ["x real bar"]
for i in range(len(bins) - 1):
    lines.append(f"{(bins[i]+bins[i+1])/2:.0f} {max(hr[i],0.5):.1f} {max(hb[i],0.5):.1f}")
w("orf_hist.dat", "\n".join(lines))
# línea teórica escalada al primer bin barajado
tt = ["x teo"]
for x in range(30, 400, 5):
    tt.append(f"{x} {hb[0]*(1-pstop)**(x-40):.3f}")
w("orf_teo.dat", "\n".join(tt))

# Modelo de codones autoentrenado (ORFs largos) y puntuación log-odds
train = [o for o in O_real if len(o) // 3 >= 300]
cc = Counter()
for o in train:
    for i in range(0, len(o), 3):
        cc[o[i:i + 3]] += 1
tot = sum(cc.values())
cod = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
pcod = {c: (cc[c] + 1) / (tot + 64) for c in cod}
qcod = {c: freq[c[0]] * freq[c[1]] * freq[c[2]] for c in cod}
llr = {c: math.log2(pcod[c] / qcod[c]) for c in cod}
say(f"[cod] entrenamiento: {len(train)} ORFs >=300 codones, {tot:,} codones")
top = sorted(cod, key=lambda c: llr[c])
say("[cod] codones más favorecidos:",
    [(c, round(llr[c], 2)) for c in top[::-1][:6]])
say("[cod] más desfavorecidos (no stop):",
    [(c, round(llr[c], 2)) for c in top if c not in STOP][:6])


def score(o):
    return np.mean([llr[o[i:i + 3]] for i in range(0, len(o), 3)])


sr = np.array([score(o) for o in O_real if 60 <= len(o) // 3 < 150])
sb = np.array([score(o) for o in O_bar if 60 <= len(o) // 3 < 150])
say(f"[cod] ORFs 60-149 codones: real n={len(sr)} barajado n={len(sb)}")
say(f"[cod] puntuación media: real={sr.mean():.3f} barajado={sb.mean():.3f};"
    f" fracción real > 0.1 = {np.mean(sr>0.1):.3f}; barajado > 0.1 = {np.mean(sb>0.1):.4f}")
bins = np.arange(-0.6, 0.62, 0.04)
hr, _ = np.histogram(sr, bins)
hb, _ = np.histogram(sb, bins)
lines = ["x real bar"]
for i in range(len(bins) - 1):
    lines.append(f"{(bins[i]+bins[i+1])/2:.3f} {hr[i]} {hb[i]}")
w("orf_score.dat", "\n".join(lines))
# ORF largo concreto: primer ORF >= 300 codones
o = train[0]
say(f"[cod] ejemplo ORF de {len(o)//3} codones: puntuación={score(o):.3f}"
    f" bits/codón; total={score(o)*len(o)//3:.1f} bits; inicio {o[:24]}")
ej_orf = "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGA"
say(f"[cod] ORF de ejemplo ({len(ej_orf)//3} codones) = {ej_orf}")
say("[cod]  log-odds por codón:",
    [(ej_orf[i:i+3], round(llr[ej_orf[i:i+3]], 2)) for i in range(0, 24, 3)])
say(f"[cod]  media={score(ej_orf[:-3]):.3f}")


# =====================================================================
# 8.3  Fracción de k-mers del genoma no observados (motivo del multi-k)
# =====================================================================
lines = ["k c5 c10 c30"]
for k in range(15, 128, 2):
    fila = [k]
    for c in (5, 10, 30):
        lam_k = c * (150 - k + 1) / 150 * (1 - 0.005) ** k
        fila.append(math.exp(-lam_k))
    lines.append(" ".join(f"{v:.6g}" for v in fila))
w("ausentes.dat", "\n".join(lines))
for c in (5, 10, 30):
    for k in (21, 33, 55, 77, 127):
        lam_k = c * (150 - k + 1) / 150 * (1 - 0.005) ** k
        say(f"[multik] c={c} k={k}: lambda_k={lam_k:.2f}"
            f"  ausentes={math.exp(-lam_k):.3g}  en 5 Mb={5e6*math.exp(-lam_k):,.0f}")

(OUT / "cifras.txt").write_text("\n".join(LOG) + "\n")
