"""Genera datos y figuras del capítulo 12 (transcriptómica de célula única).

Ejecutar desde libro/:  python3 figuras/cap12/generar.py
Requiere numpy, scipy, scikit-learn, networkx, matplotlib y umap-learn.
Todos los datos son SIMULADOS (no se descarga ningún conjunto de datos).
Escribe .dat/.tex/.pdf en figuras/cap12/ e imprime (y guarda en cifras.txt)
las cifras que se citan en el texto: única fuente de verdad.
"""
from pathlib import Path
import math
import numpy as np
from scipy import stats, sparse
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
import networkx as nx
import umap

import matplotlib
matplotlib.use("pdf")
from matplotlib import font_manager, pyplot as plt, rcParams
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter

OUT = Path(__file__).parent
CIFRAS = []


def w(name, text):
    (OUT / name).write_text(text)


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    CIFRAS.append(s)


# ---------------------------------------------------------------- estilo mpl
FD = "/usr/local/texlive/2025/texmf-dist/fonts/opentype/adobe/sourcesanspro/"
for f in ("SourceSansPro-Regular.otf", "SourceSansPro-Bold.otf",
          "SourceSansPro-RegularIt.otf"):
    try:
        font_manager.fontManager.addfont(FD + f)
    except Exception as e:  # pragma: no cover
        print("fuente:", e)
rcParams.update({
    "font.family": "Source Sans Pro", "font.size": 8.5,
    "pdf.fonttype": 3, "mathtext.fontset": "custom",
    "mathtext.rm": "Source Sans Pro", "mathtext.it": "Source Sans Pro:italic",
    "axes.edgecolor": "#C3C2B7", "axes.linewidth": 0.6,
    "axes.labelcolor": "#52514E", "xtick.color": "#898781",
    "ytick.color": "#898781", "xtick.labelcolor": "#898781",
    "ytick.labelcolor": "#898781", "axes.spines.top": False,
    "axes.spines.right": False, "savefig.transparent": True,
    "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.titlecolor": "#0B0B0B", "legend.frameon": False,
    "axes.unicode_minus": False,
})
C = dict(azul="#2A78D6", naranja="#EB6834", aqua="#1BAF7A",
         amarillo="#EDA100", magenta="#E87BA4", verde="#008300",
         violeta="#4A3AA7", rojo="#E34948", tinta="#0B0B0B",
         tinta2="#52514E", gris="#898781", rejilla="#E1E0D9",
         base="#C3C2B7", papel="#F0EFEC", azulnoche="#0D366B",
         azulprofundo="#104281", azulclaro="#9EC5F4", rojooscuro="#B8302F")
PAL = [C["azul"], C["aqua"], C["violeta"], C["naranja"], C["rojo"],
       C["magenta"], C["amarillo"], C["verde"], C["azulnoche"], C["gris"],
       C["azulclaro"], C["rojooscuro"]]
CM = 1 / 2.54
coma = FuncFormatter(lambda v, p: f"{v:g}".replace(".", ","))


def fmt(ax, x=True, y=True):
    if x:
        ax.xaxis.set_major_formatter(coma)
    if y:
        ax.yaxis.set_major_formatter(coma)


def letra(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=11,
            fontweight="bold", color=C["tinta"], va="bottom")


def limpio(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def guardar(fig, nombre):
    fig.savefig(OUT / nombre, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


# =====================================================================
# Simulador de cuentas de UMI (gamma-Poisson = binomial negativa)
# =====================================================================
MT = ["MT-ND1", "MT-ND2", "MT-CO1", "MT-CO2", "MT-ATP8", "MT-ATP6",
      "MT-CO3", "MT-ND3", "MT-ND4L", "MT-ND4", "MT-ND5", "MT-ND6", "MT-CYB"]
TIPOS = ["T CD4", "T CD8", "NK", "B", "Mono CD14", "Mono FCGR3A", "DC"]
PROP = np.array([0.30, 0.15, 0.10, 0.14, 0.18, 0.08, 0.05])
LIBF = np.array([0.9, 0.9, 1.0, 1.0, 1.45, 1.3, 1.3])  # tamaño relativo
# marcadores: gen -> {tipo: log2 FC}
MARC = {
    "CD3E": {0: 4, 1: 4}, "CD3D": {0: 4, 1: 4}, "IL7R": {0: 4, 1: 1.5},
    "CCR7": {0: 3.5}, "CD4": {0: 2.5, 4: 2, 5: 2, 6: 2},
    "CD8A": {1: 5}, "CD8B": {1: 4.5}, "GZMK": {1: 4, 2: 1.5},
    "NKG7": {2: 6, 1: 3}, "GNLY": {2: 6.5}, "KLRD1": {2: 4.5},
    "PRF1": {2: 4.5, 1: 1.5}, "GZMB": {2: 5},
    "MS4A1": {3: 5.5}, "CD79A": {3: 5.5}, "CD79B": {3: 4.5},
    "BANK1": {3: 4}, "CD19": {3: 3.5},
    "CD14": {4: 5, 5: 1.5}, "LYZ": {4: 6.5, 5: 4.5, 6: 5},
    "S100A8": {4: 6.5, 5: 2.5}, "S100A9": {4: 6.5, 5: 3.5, 6: 2},
    "VCAN": {4: 4.5}, "FCGR3A": {5: 5.5, 2: 3.5}, "MS4A7": {5: 4.5},
    "LST1": {5: 4, 4: 2}, "CDKN1C": {5: 4},
    "FCER1A": {6: 5}, "CST3": {6: 5, 4: 4, 5: 4}, "CLEC10A": {6: 4.5},
    "CD1C": {6: 4.5}, "HLA-DQA1": {6: 4.5, 3: 3},
}


import os
FCG = float(os.environ.get("FCG", 0.5))
FCT = float(os.environ.get("FCT", 0.55))
ESCM = float(os.environ.get("ESCM", 0.7))
GRAD = float(os.environ.get("GRAD", 2.0))


def construir_perfiles(G=2000, seed=11):
    rng = np.random.default_rng(seed)
    nombres = MT + list(MARC)
    nombres += [f"G{i:04d}" for i in range(G - len(nombres))]
    G = len(nombres)
    base = rng.lognormal(0.0, 1.3, G)
    im = {g: i for i, g in enumerate(nombres)}
    for g in MARC:
        base[im[g]] = rng.lognormal(-1.2, 0.3)
    base[:len(MT)] = 0.0
    lf = np.zeros((len(TIPOS), G))
    otros = np.arange(len(MT) + len(MARC), G)
    grupos = {"linfoide": [0, 1, 2, 3], "TNK": [0, 1, 2],
              "mieloide": [4, 5, 6], "mono": [4, 5]}
    for gname, ts in grupos.items():
        idx = rng.choice(otros, 60, replace=False)
        fc = rng.normal(FCG, 0.3, 60) * rng.choice([-1, 1], 60, p=[.3, .7])
        for t in ts:
            lf[t, idx] += fc
    for t in range(len(TIPOS)):
        idx = rng.choice(otros, 30, replace=False)
        lf[t, idx] += rng.normal(FCT, 0.3, 30) * rng.choice([-1, 1], 30)
    for g, d in MARC.items():
        for t, v in d.items():
            lf[t, im[g]] += ESCM * v * math.log(2)
    P = base[None, :] * np.exp(lf)
    P /= P.sum(1, keepdims=True)
    mt_perfil = rng.dirichlet(np.full(len(MT), 5.0))
    theta = rng.lognormal(math.log(6), 0.5, G)
    # gradiente continuo dentro de T CD4 (naive -> memoria)
    grad_idx = rng.choice(otros, 40, replace=False)
    grad_fc = rng.normal(GRAD, 0.3, 40) * rng.choice([-1, 1], 40)
    return nombres, P, mt_perfil, theta, grad_idx, grad_fc


NOMBRES, PERF, MTP, THETA, GIDX, GFC = construir_perfiles()
IM = {g: i for i, g in enumerate(NOMBRES)}
NG = len(NOMBRES)
NMT = len(MT)
AMB = (PROP * LIBF) @ PERF          # perfil ambiental ~ mezcla lisada
AMB /= AMB.sum()


def nb(rng, mu, theta):
    return rng.poisson(rng.gamma(theta, mu / theta))


def celulas(rng, tipos, L, fmt_mt, amb=0.02, fc_lote=None):
    """Matriz de cuentas (células x genes) para los tipos dados."""
    n = len(tipos)
    X = np.zeros((n, NG), dtype=np.int32)
    for i in range(n):
        p = PERF[tipos[i]].copy()
        if tipos[i] == 0:                     # estado continuo en T CD4
            p[GIDX] *= np.exp(GFC * rng.uniform(0, 1))
            p /= p.sum()
        if fc_lote is not None:
            p = p * fc_lote
            p /= p.sum()
        p = (1 - fmt_mt[i]) * p
        p[:NMT] = fmt_mt[i] * MTP
        p = (1 - amb) * p + amb * AMB
        X[i] = nb(rng, L[i] * p, THETA)
    return X


def sim_pbmc(n=3000, seed=2025, n_mueren=150, n_dob=210, fc_lote=None):
    rng = np.random.default_rng(seed)
    t = rng.choice(len(TIPOS), n, p=PROP)
    L = rng.lognormal(math.log(2600), 0.42, n) * LIBF[t]
    fmt_mt = rng.beta(8, 190, n)
    X = celulas(rng, t, L, fmt_mt, fc_lote=fc_lote)
    estado = np.array(["ok"] * n, dtype=object)
    if n_mueren:
        tm = rng.choice(len(TIPOS), n_mueren, p=PROP)
        Lm = rng.lognormal(math.log(900), 0.6, n_mueren)
        fm = rng.uniform(0.18, 0.65, n_mueren)
        Xm = celulas(rng, tm, Lm, fm)
        X = np.vstack([X, Xm]); t = np.r_[t, tm]
        estado = np.r_[estado, ["baja"] * n_mueren]
    if n_dob:
        a = rng.choice(len(TIPOS), n_dob, p=PROP)
        b = rng.choice(len(TIPOS), n_dob, p=PROP)
        La = rng.lognormal(math.log(2600), 0.42, n_dob) * LIBF[a]
        Lb = rng.lognormal(math.log(2600), 0.42, n_dob) * LIBF[b]
        Xd = (celulas(rng, a, La, rng.beta(8, 190, n_dob))
              + celulas(rng, b, Lb, rng.beta(8, 190, n_dob)))
        X = np.vstack([X, Xd]); t = np.r_[t, a]
        estado = np.r_[estado, np.where(a == b, "dob_homo", "dob_hetero")]
    return X, t, estado


# ---------------------------------------------------------------- utilidades
def lognorm(X, objetivo=1e4):
    tot = X.sum(1, keepdims=True)
    return np.log1p(X / tot * objetivo)


def hvg(Y, n_top=500, nbins=20):
    """Dispersión normalizada por intervalos de media (sabor 'seurat')."""
    E = np.expm1(Y)
    mu = E.mean(0)
    var = E.var(0, ddof=1)
    disp = np.log(np.where(mu > 0, var / np.maximum(mu, 1e-12), 1e-12))
    lmu = np.log1p(mu)
    bins = np.quantile(lmu[mu > 0], np.linspace(0, 1, nbins + 1))
    z = np.full(len(mu), -np.inf)
    lab = np.clip(np.digitize(lmu, bins[1:-1]), 0, nbins - 1)
    for b in range(nbins):
        m = (lab == b) & (mu > 0)
        if m.sum() > 1:
            z[m] = (disp[m] - disp[m].mean()) / (disp[m].std() + 1e-9)
    z[:NMT] = -np.inf
    top = np.argsort(-z)[:n_top]
    return top, lmu, z


def pca(Y, genes, k=50):
    Z = Y[:, genes]
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    Z = np.clip(Z, -10, 10)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    pcs = U[:, :k] * S[:k]
    ev = S ** 2 / (Z.shape[0] - 1)
    return pcs, ev[:k] / ev.sum(), Vt[:k]


def knn_grafo(Z, k=15):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    d, idx = nn.kneighbors(Z)
    n = Z.shape[0]
    filas = np.repeat(np.arange(n), k)
    A = sparse.csr_matrix((np.ones(n * k), (filas, idx[:, 1:].ravel())),
                          shape=(n, n))
    A = ((A + A.T) > 0).astype(float)
    return A, idx[:, 1:], d[:, 1:]


def louvain(A, res=1.0, seed=0):
    Gx = nx.from_scipy_sparse_array(A)
    com = nx.community.louvain_communities(Gx, resolution=res, seed=seed)
    lab = np.zeros(A.shape[0], int)
    com = sorted(com, key=len, reverse=True)
    for c, s in enumerate(com):
        lab[list(s)] = c
    Q = nx.community.modularity(Gx, com, resolution=res)
    Q1 = nx.community.modularity(Gx, com, resolution=1.0)
    desc = sum(1 for s in com if len(s) > 1
               and not nx.is_connected(Gx.subgraph(s)))
    return lab, Q, Q1, desc


# =====================================================================
# 12.1  Gotas, códigos de barras, QC y normalización
# =====================================================================
say("=== 12.1 ===")
for lam in (0.05, 0.1, 0.2, 0.5):
    p0 = math.exp(-lam); p1 = lam * p0
    dob = (1 - p0 - p1) / (1 - p0)
    say(f"[poisson] lambda={lam}: P(vacia)={p0:.3f} P(1)={p1:.4f} "
        f"P(>=2|>=1)={dob:.4f}")
say(f"[barcode] 4^16={4**16:.3e}  4^12={4**12:.3e}  4^10={4**10:.3e}")
for n_mol in (10, 100, 1000):
    say(f"[umi] n={n_mol}: colisiones esperadas 12nt="
        f"{n_mol*(n_mol-1)/2/4**12:.4f}  10nt={n_mol*(n_mol-1)/2/4**10:.4f}")

Xall, tall, est = sim_pbmc()
N0 = Xall.shape[0]
say(f"[sim] codigos con celula: {N0} (ok={np.sum(est=='ok')}, "
    f"baja={np.sum(est=='baja')}, dobletes={np.sum(est!='ok')-np.sum(est=='baja')})")
say(f"[sim] genes: {NG}; mitocondriales: {NMT}")

# ---- gotas vacías y curva de rodilla
rng = np.random.default_rng(7)
n_vac = 60000
Lv = rng.lognormal(math.log(12), 0.9, n_vac)
tot_vac = rng.poisson(Lv)
tot_cel = Xall.sum(1)
tot = np.sort(np.r_[tot_cel, tot_vac])[::-1]
rank = np.arange(1, len(tot) + 1)
ok = tot > 0
lr, lt = np.log10(rank[ok]), np.log10(tot[ok])
grid = np.linspace(0, lr.max(), 300)
lts = np.interp(grid, lr, lt)
sl = np.gradient(lts, grid)
sl_s = np.convolve(sl, np.ones(7) / 7, mode="same")
i_inf = np.argmin(sl_s[20:-20]) + 20
r_inf, t_inf = 10 ** grid[i_inf], 10 ** lts[i_inf]
say(f"[rodilla] inflexion en rango ~{r_inf:.0f} con {t_inf:.0f} UMI; "
    f"codigos con >=100 UMI: {np.sum(tot>=100)}; total codigos {len(tot)}")
sel = np.unique(np.r_[np.round(np.logspace(0, np.log10(ok.sum()), 260)).astype(int) - 1])
es_cel = np.zeros(len(tot), bool)
w("rodilla.dat", "rango umi\n" + "\n".join(
    f"{rank[i]} {tot[i]}" for i in sel if tot[i] > 0))
w("rodilla_inf.tex", f"\\def\\capdocerinf{{{r_inf:.0f}}}\\def\\capdocetinf{{{t_inf:.0f}}}\n")

# ---- matriz dispersa
Xs = sparse.csr_matrix(Xall)
dens = Xs.nnz / np.prod(Xs.shape)
by_dense = np.prod(Xs.shape) * 4
by_csr = Xs.nnz * (4 + 4) + (Xs.shape[0] + 1) * 8
say(f"[dispersa] forma={Xs.shape} nnz={Xs.nnz} densidad={dens:.4f} "
    f"densa={by_dense/1e6:.1f} MB csr={by_csr/1e6:.1f} MB "
    f"ahorro={by_dense/by_csr:.1f}x")
n1, g1 = 1_000_000, 30_000
say(f"[dispersa] 1e6x3e4 densa float32={n1*g1*4/1e9:.0f} GB;"
    f" csr al {dens*100:.1f}%={n1*g1*dens*8/1e9:.1f} GB")
say("[dispersa] ejemplo 4x6 no nulo -> ver texto")
M = np.array([[0, 3, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0],
              [2, 0, 0, 5, 0, 0], [0, 1, 0, 0, 0, 7]])
Mc = sparse.csr_matrix(M)
say(f"[csr] data={Mc.data.tolist()} indices={Mc.indices.tolist()} "
    f"indptr={Mc.indptr.tolist()}")

# ---- métricas QC
tc = Xall.sum(1)
ng = (Xall > 0).sum(1)
pmt = 100 * Xall[:, :NMT].sum(1) / tc


def mad(x):
    return np.median(np.abs(x - np.median(x))) * 1.4826


ltc, lng = np.log1p(tc), np.log1p(ng)
lo_tc = np.expm1(np.median(ltc) - 5 * mad(ltc))
hi_tc = np.expm1(np.median(ltc) + 5 * mad(ltc))
lo_ng = np.expm1(np.median(lng) - 5 * mad(lng))
hi_ng = np.expm1(np.median(lng) + 5 * mad(lng))
mt_thr = max(np.median(pmt) + 3 * mad(pmt), 8.0)
say(f"[qc] mediana UMI={np.median(tc):.0f} genes={np.median(ng):.0f} "
    f"%mt={np.median(pmt):.2f}; MAD(log UMI)={mad(ltc):.3f}")
say(f"[qc] umbrales: UMI [{lo_tc:.0f},{hi_tc:.0f}] genes [{lo_ng:.0f},"
    f"{hi_ng:.0f}] %mt<{mt_thr:.2f} (mediana+3MAD={np.median(pmt)+3*mad(pmt):.2f})")
keep = ((tc >= lo_tc) & (tc <= hi_tc) & (ng >= lo_ng) & (ng <= hi_ng)
        & (pmt < mt_thr))
for e in ("ok", "baja", "dob_hetero", "dob_homo"):
    m = est == e
    say(f"[qc] {e}: n={m.sum()} descartadas={np.sum(~keep & m)} "
        f"({100*np.mean(~keep[m]):.1f}%)")
say(f"[qc] retenidas {keep.sum()} de {N0}")
w("qc_umbrales.tex",
  f"\\def\\capdocemt{{{mt_thr:.1f}}}\\def\\capdocelong{{{lo_ng:.0f}}}\n")

fig, axs = plt.subplots(1, 3, figsize=(13 * CM, 5.4 * CM))
rngv = np.random.default_rng(3)
vals = [(ng, "genes detectados", [lo_ng, hi_ng], True),
        (tc, "UMI totales", [lo_tc, hi_tc], True),
        (pmt, "% UMI mitocondriales", [mt_thr], False)]
for ax, (v, tit, thr, lg), le in zip(axs, vals, "abc"):
    vv = np.log10(v) if lg else v
    parts = ax.violinplot(vv, positions=[0], widths=0.9, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(C["azulclaro"]); b.set_edgecolor(C["azul"])
        b.set_alpha(0.55); b.set_linewidth(0.6)
    jit = rngv.uniform(-0.33, 0.33, len(v))
    col = np.where(keep, C["tinta2"], C["rojo"])
    ax.scatter(jit[keep], vv[keep], s=0.6, c=C["tinta2"], alpha=0.35, lw=0)
    ax.scatter(jit[~keep], vv[~keep], s=2.2, c=C["rojo"], alpha=0.9, lw=0)
    for t in thr:
        tt = np.log10(t) if lg else t
        ax.axhline(tt, color=C["rojo"], ls=(0, (3, 2)), lw=0.9)
    ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
    ax.set_title(tit, fontsize=8.5)
    if lg:
        tk = [t for t in (100, 300, 1000, 3000, 10000, 30000)
              if np.log10(t) >= vv.min() - .1 and np.log10(t) <= vv.max() + .1]
        ax.set_yticks(np.log10(tk))
        ax.set_yticklabels([f"{t:,}".replace(",", " ") for t in tk])
    else:
        fmt(ax, x=False)
    letra(ax, le, x=-0.3)
fig.tight_layout(w_pad=1.2)
guardar(fig, "qc_violines.pdf")

# ---- dobletes estilo Scrublet sobre las células que pasan QC
Xq, tq, eq = Xall[keep], tall[keep], est[keep]
rngd = np.random.default_rng(5)
Yq = lognorm(Xq)
gq, _, _ = hvg(Yq, 500)
Zo = Yq[:, gq]
mu_, sd_ = Zo.mean(0), Zo.std(0) + 1e-9
Zs = (Zo - mu_) / sd_
U, S, Vt = np.linalg.svd(Zs, full_matrices=False)
Po = Zs @ Vt[:30].T
nsim = 2 * len(Xq)
i1, i2 = rngd.integers(0, len(Xq), nsim), rngd.integers(0, len(Xq), nsim)
Xsim = Xq[i1] + Xq[i2]
Ps = ((lognorm(Xsim)[:, gq] - mu_) / sd_) @ Vt[:30].T
kd = int(round(0.5 * math.sqrt(len(Xq))))
nnd = NearestNeighbors(n_neighbors=kd).fit(np.vstack([Po, Ps]))
_, idd = nnd.kneighbors(Po)
q = (idd >= len(Po)).mean(1)
r = nsim / len(Po)
rho = 0.07
score = q * rho / r / (1 - rho - q * (1 - rho - rho / r))
_, idsim = nnd.kneighbors(Ps)
qs = (idsim[:, 1:] >= len(Po)).mean(1)
ssim = qs * rho / r / (1 - rho - qs * (1 - rho - rho / r))
thr_d = 0.25
esdob = np.isin(eq, ["dob_hetero", "dob_homo"])
say(f"[scrublet] k={kd} r={r:.1f} umbral={thr_d}: "
    f"heterotipicos detectados {np.mean(score[eq=='dob_hetero']>thr_d)*100:.1f}% "
    f"homotipicos {np.mean(score[eq=='dob_homo']>thr_d)*100:.1f}% "
    f"singletes marcados {np.mean(score[eq=='ok']>thr_d)*100:.2f}%")
say(f"[scrublet] mediana puntuacion singletes={np.median(score[eq=='ok']):.3f}"
    f" dobletes hetero={np.median(score[eq=='dob_hetero']):.3f}")
bins = np.linspace(0, 1, 41)
h_ok = np.histogram(np.clip(score[~esdob], 0, 1), bins)[0]
h_db = np.histogram(np.clip(score[esdob], 0, 1), bins)[0]
h_sim = np.histogram(np.clip(ssim, 0, 1), bins)[0]
w("dobletes.dat", "x ok dob sim\n" + "\n".join(
    f"{(bins[i]+bins[i+1])/2:.4f} {h_ok[i]} {h_db[i]} {h_sim[i]/r:.2f}"
    for i in range(40)))
w("dobletes_thr.tex", f"\\def\\capdocethrd{{{thr_d}}}\n")

keep2 = score <= thr_d
X, T, E = Xq[keep2], tq[keep2], eq[keep2]
say(f"[limpio] tras QC+dobletes: {len(X)} celulas; "
    f"quedan dobletes reales={np.sum(E!='ok')}")

# ---- ARN ambiental: LYZ en células B
iB = T == 3
lyzB = np.mean(X[iB, IM["LYZ"]] > 0)
lyzB_sin = None
say(f"[ambiental] fraccion de celulas B con LYZ>0: {lyzB*100:.1f}%;"
    f" UMI medios de LYZ en B={X[iB, IM['LYZ']].mean():.2f} vs "
    f"Mono CD14={X[T==4, IM['LYZ']].mean():.1f}")

# ---- normalización y relación media-varianza
tot = X.sum(1)
sfac = tot / tot.mean()
Xn = X / sfac[:, None]
mu_g = Xn.mean(0); var_g = Xn.var(0, ddof=1)
okg = mu_g > 1e-3
ok2 = okg & (np.arange(NG) >= NMT + len(MARC))
# theta común por momentos dentro de un tipo (T CD4) sobre genes de fondo
i4 = T == 0
Xc = X[i4] / (X[i4].sum(1) / X[i4].sum(1).mean())[:, None]
m4, v4 = Xc.mean(0), Xc.var(0, ddof=1)
g4 = (m4 > 0.05) & (np.arange(NG) >= NMT + len(MARC))
th_hat = np.median(m4[g4] ** 2 / np.maximum(v4[g4] - m4[g4], 1e-6))
say(f"[nb] theta mediano (momentos, T CD4, genes de fondo)={th_hat:.2f}; "
    f"theta simulado mediano={np.median(THETA):.2f}")
Y = lognorm(X)
hv, lmu, zd = hvg(Y, 500)
es_hv = np.zeros(NG, bool); es_hv[hv] = True
nm_hv = [NOMBRES[i] for i in hv if NOMBRES[i] in MARC]
say(f"[hvg] {len(hv)} HVG; marcadores nombrados entre HVG: {len(nm_hv)}/"
    f"{len(MARC)}; MT entre HVG: {np.sum(hv<NMT)}")
say(f"[hvg] top10: {[NOMBRES[i] for i in hv[:10]]}")
# ejemplo numérico de normalización
ej = np.array([X[np.argmax(T == 4), IM['LYZ']], X[np.argmax(T == 0), IM['LYZ']]])
c0 = np.argmax(T == 4)
say(f"[norm] celula0 (Mono CD14): total={tot[c0]} LYZ={X[c0, IM['LYZ']]} "
    f"CP10k={X[c0, IM['LYZ']]/tot[c0]*1e4:.1f} log1p={Y[c0, IM['LYZ']]:.3f}")

fig, axs = plt.subplots(1, 2, figsize=(13 * CM, 5.6 * CM))
ax = axs[0]
mm = okg
ax.scatter(mu_g[mm & ~es_hv], var_g[mm & ~es_hv], s=2, c=C["base"], lw=0)
ax.scatter(mu_g[mm & es_hv], var_g[mm & es_hv], s=3, c=C["naranja"], lw=0,
           label="HVG")
xx = np.logspace(-3, np.log10(mu_g.max()) + .2, 100)
ax.plot(xx, xx, color=C["azul"], lw=1.1, label="Poisson: $\\sigma^2=\\mu$")
ax.plot(xx, xx + xx ** 2 / th_hat, color=C["aqua"], lw=1.1, ls="--",
        label=f"BN: $\\sigma^2=\\mu+\\mu^2/\\theta$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("media de UMI normalizados"); ax.set_ylabel("varianza")
ax.legend(fontsize=7, loc="upper left", handlelength=1.6)
letra(ax, "a")
ax = axs[1]
fin = np.isfinite(zd) & okg
ax.scatter(lmu[fin & ~es_hv], zd[fin & ~es_hv], s=2, c=C["base"], lw=0)
ax.scatter(lmu[fin & es_hv], zd[fin & es_hv], s=3, c=C["naranja"], lw=0)
for g in ("LYZ", "GNLY", "MS4A1", "CD8A", "FCER1A", "CCR7"):
    i = IM[g]
    ax.annotate(g, (lmu[i], zd[i]), fontsize=6.5, color=C["tinta2"],
                xytext=(3, 1), textcoords="offset points")
ax.axhline(np.sort(zd[fin])[::-1][499], color=C["rojo"], ls=(0, (3, 2)), lw=0.8)
ax.set_xlabel("log(1 + media)"); ax.set_ylabel("dispersión normalizada (z)")
fmt(ax)
letra(ax, "b")
fig.tight_layout(w_pad=1.5)
guardar(fig, "media_varianza.pdf")

# =====================================================================
# 12.2  PCA, t-SNE y UMAP
# =====================================================================
say("=== 12.2 ===")
pcs, frac, Vt = pca(Y, hv, 50)
say(f"[pca] var. explicada PC1..5 = {np.round(frac[:5]*100,2).tolist()} %; "
    f"acumulada 10={frac[:10].sum()*100:.1f}% 30={frac[:30].sum()*100:.1f}%")
w("pca_var.dat", "pc frac acum\n" + "\n".join(
    f"{i+1} {frac[i]*100:.4f} {frac[:i+1].sum()*100:.3f}" for i in range(50)))
# null: permutar columnas
rngp = np.random.default_rng(9)
Yp = Y[:, hv].copy()
for j in range(Yp.shape[1]):
    Yp[:, j] = rngp.permutation(Yp[:, j])
Zp = (Yp - Yp.mean(0)) / (Yp.std(0) + 1e-9)
Sp = np.linalg.svd(Zp, compute_uv=False)
fp = Sp ** 2 / (Sp ** 2).sum()
say(f"[pca] nulo permutado PC1={fp[0]*100:.2f}%; PCs reales sobre el nulo:"
    f" {int(np.sum(frac > fp[0]))}")
w("pca_nulo.dat", "pc frac\n" + "\n".join(
    f"{i+1} {fp[i]*100:.4f}" for i in range(50)))
top_pc1 = np.argsort(-np.abs(Vt[0]))[:6]
say(f"[pca] cargas mayores PC1: {[NOMBRES[hv[i]] for i in top_pc1]}")
top_pc2 = np.argsort(-np.abs(Vt[1]))[:6]
say(f"[pca] cargas mayores PC2: {[NOMBRES[hv[i]] for i in top_pc2]}")
Z30 = pcs[:, :30]

emb = {}
emb["PCA"] = pcs[:, :2]
emb["t-SNE"] = TSNE(perplexity=30, init="pca", random_state=0).fit_transform(Z30)
emb["UMAP"] = umap.UMAP(n_neighbors=15, min_dist=0.3,
                        random_state=0).fit_transform(Z30)
perp = {}
for p in (5, 30, 300):
    perp[p] = emb["t-SNE"] if p == 30 else TSNE(
        perplexity=p, init="pca", random_state=0).fit_transform(Z30)


def panel_tipos(ax, E2, t, s=1.2, leyenda=False):
    orden = np.random.default_rng(0).permutation(len(t))
    ax.scatter(E2[orden, 0], E2[orden, 1], s=s, lw=0,
               c=[PAL[i] for i in t[orden]])
    limpio(ax)
    if leyenda:
        for i, nmb in enumerate(TIPOS):
            ax.scatter([], [], s=12, c=PAL[i], label=nmb)


fig, axs = plt.subplots(1, 3, figsize=(13 * CM, 4.8 * CM))
for ax, k, le in zip(axs, ("PCA", "t-SNE", "UMAP"), "abc"):
    panel_tipos(ax, emb[k], T, leyenda=(k == "UMAP"))
    ax.set_title(k)
    letra(ax, le, x=-0.02)
    lbl = {"PCA": ("PC1", "PC2"), "t-SNE": ("t-SNE 1", "t-SNE 2"),
           "UMAP": ("UMAP 1", "UMAP 2")}[k]
    ax.set_xlabel(lbl[0], fontsize=7); ax.set_ylabel(lbl[1], fontsize=7)
fig.legend(*axs[2].get_legend_handles_labels(), loc="lower center", ncol=7,
           fontsize=7, bbox_to_anchor=(0.5, -0.08), handletextpad=0.1,
           columnspacing=0.8)
fig.tight_layout()
guardar(fig, "embeddings.pdf")

fig, axs = plt.subplots(1, 3, figsize=(13 * CM, 4.6 * CM))
for ax, p, le in zip(axs, (5, 30, 300), "abc"):
    panel_tipos(ax, perp[p], T)
    ax.set_title(f"perplejidad = {p}")
    letra(ax, le, x=-0.02)
fig.tight_layout()
guardar(fig, "perplejidad.pdf")

# preservación de distancias
rngd2 = np.random.default_rng(4)
npair = 4000
a_, b_ = rngd2.integers(0, len(Z30), npair), rngd2.integers(0, len(Z30), npair)
m_ = a_ != b_
a_, b_ = a_[m_], b_[m_]
dh = np.linalg.norm(Z30[a_] - Z30[b_], axis=1)
fig, axs = plt.subplots(1, 3, figsize=(13 * CM, 4.6 * CM), sharex=True)
rho_d = {}
for ax, k, le in zip(axs, ("PCA", "t-SNE", "UMAP"), "abc"):
    dl = np.linalg.norm(emb[k][a_] - emb[k][b_], axis=1)
    rho_d[k] = stats.spearmanr(dh, dl).statistic
    ax.scatter(dh, dl, s=1, c=C["azul"], alpha=0.25, lw=0)
    ax.set_title(f"{k}  ($\\rho$ = {rho_d[k]:.2f})".replace(".", ","))
    ax.set_xlabel("distancia en 30 PC")
    if k == "PCA":
        ax.set_ylabel("distancia en 2D")
    fmt(ax)
    letra(ax, le, x=-0.25)
# kNN preservados
def knn_set(Zx, k=15):
    return NearestNeighbors(n_neighbors=k + 1).fit(Zx).kneighbors(Zx)[1][:, 1:]
ref = knn_set(Z30)
presk = {}
for k in ("PCA", "t-SNE", "UMAP"):
    kk = knn_set(emb[k])
    presk[k] = np.mean([len(set(ref[i]) & set(kk[i])) / 15
                        for i in range(len(ref))])
say("[distancias] Spearman " + ", ".join(f"{k}={v:.3f}" for k, v in rho_d.items()))
say("[distancias] fraccion kNN(15) conservados " +
    ", ".join(f"{k}={v:.3f}" for k, v in presk.items()))
# distancias entre centroides de tipos
cent_h = np.array([Z30[T == t].mean(0) for t in range(7)])
for k in ("t-SNE", "UMAP"):
    ce = np.array([emb[k][T == t].mean(0) for t in range(7)])
    iu = np.triu_indices(7, 1)
    dH = np.linalg.norm(cent_h[:, None] - cent_h[None], axis=2)[iu]
    dE = np.linalg.norm(ce[:, None] - ce[None], axis=2)[iu]
    say(f"[distancias] centroides {k}: Spearman={stats.spearmanr(dH, dE).statistic:.3f}")
fig.tight_layout(w_pad=1.0)
guardar(fig, "distancias.pdf")

# ejemplo t-SNE: perplejidad -> sigma para una célula
def sigma_perp(d2, perp_obj, tol=1e-5):
    lo, hi = 1e-6, 1e6
    for _ in range(200):
        s = math.sqrt(lo * hi)
        p = np.exp(-(d2 - d2.min()) / (2 * s * s)); p /= p.sum()
        H = -np.sum(p[p > 0] * np.log2(p[p > 0]))
        if abs(2 ** H - perp_obj) < tol:
            break
        if 2 ** H > perp_obj:
            hi = s
        else:
            lo = s
    return s, p


d2 = np.sum((Z30[1:] - Z30[0]) ** 2, axis=1)
for pp in (5, 30, 300):
    s_, p_ = sigma_perp(d2, pp)
    nef = 1 / np.sum(p_ ** 2)
    say(f"[tsne] celula 0: perplejidad {pp} -> sigma={s_:.3f}, "
        f"max p_j|i={p_.max():.3f}, vecinos con p>1/N: {np.sum(p_>1/len(p_))}")

# =====================================================================
# 12.3  Grafo kNN, Louvain, marcadores, lotes
# =====================================================================
say("=== 12.3 ===")
A, knn_idx, knn_d = knn_grafo(Z30, 15)
say(f"[knn] aristas={int(A.nnz/2)} grado medio={A.sum(1).mean():.1f}")
res_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0]
filas = []
labs = {}
for rres in res_list:
    lab, Q, Q1, desc = louvain(A, rres, seed=0)
    ari = adjusted_rand_score(T, lab)
    labs[rres] = lab
    filas.append((rres, lab.max() + 1, Q1, ari, desc))
    say(f"[louvain] res={rres}: k={lab.max()+1} Q(1)={Q1:.3f} ARI={ari:.3f}"
        f" comunidades desconectadas={desc}")
w("resolucion.dat", "res k Q ari\n" + "\n".join(
    f"{a} {b} {c:.4f} {d:.4f}" for a, b, c, d, _ in filas))
RES = 1.0
lab = labs[RES]
K = lab.max() + 1
# tabla de contingencia
tab = np.zeros((K, 7), int)
for c, t in zip(lab, T):
    tab[c, t] += 1
anot = [TIPOS[np.argmax(tab[c])] for c in range(K)]
pur = [tab[c].max() / tab[c].sum() for c in range(K)]
for c in range(K):
    say(f"[clusters] c{c}: n={tab[c].sum()} -> {anot[c]} (pureza {pur[c]*100:.1f}%)")

# grafo kNN dibujado sobre UMAP
fig, ax = plt.subplots(figsize=(8.5 * CM, 7.2 * CM))
U2 = emb["UMAP"]
Ac = sparse.triu(A).tocoo()
segs = np.stack([U2[Ac.row], U2[Ac.col]], axis=1)
dseg = np.linalg.norm(segs[:, 0] - segs[:, 1], axis=1)
mismo = lab[Ac.row] == lab[Ac.col]
ax.add_collection(LineCollection(segs[mismo], colors=C["base"],
                                 linewidths=0.15, alpha=0.5, zorder=1))
ax.add_collection(LineCollection(segs[~mismo], colors=C["rojo"],
                                 linewidths=0.12, alpha=0.07, zorder=1))
orden = np.random.default_rng(0).permutation(len(lab))
ax.scatter(U2[orden, 0], U2[orden, 1], s=2.5, lw=0,
           c=[PAL[c % len(PAL)] for c in lab[orden]], zorder=2)
for c in range(K):
    mc = np.median(U2[lab == c], axis=0)
    ax.text(mc[0], mc[1], f"{c}", fontsize=8, fontweight="bold",
            color=C["tinta"], ha="center", va="center", zorder=3,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                      alpha=0.8))
limpio(ax)
guardar(fig, "grafo_knn.pdf")
say(f"[knn] aristas entre clusters: {np.sum(~mismo)} de {len(mismo)} "
    f"({100*np.mean(~mismo):.1f}%)")

# Wilcoxon uno contra el resto
def wilcoxon_1vr(Yg, lab, c):
    a = Yg[lab == c]; b = Yg[lab != c]
    n1, n2 = len(a), len(b)
    R = stats.rankdata(np.vstack([a, b]), axis=0)
    R1 = R[:n1].sum(0)
    U1 = R1 - n1 * (n1 + 1) / 2
    auc = U1 / (n1 * n2)
    # varianza con corrección por empates
    Nt = n1 + n2
    tie = np.array([np.sum(np.unique(R[:, j], return_counts=True)[1] ** 3
                           - np.unique(R[:, j], return_counts=True)[1])
                    for j in range(R.shape[1])])
    varU = n1 * n2 / 12 * ((Nt + 1) - tie / (Nt * (Nt - 1)))
    z = (U1 - n1 * n2 / 2) / np.sqrt(np.maximum(varU, 1e-12))
    p = 2 * stats.norm.sf(np.abs(z))
    lfc = np.log2(np.expm1(a).mean(0) + 1e-9) - np.log2(np.expm1(b).mean(0) + 1e-9)
    return z, p, auc, lfc


gsel = np.where(Y.sum(0) > 0)[0]
top_marc = {}
for c in range(K):
    z, p, auc, lfc = wilcoxon_1vr(Y[:, gsel], lab, c)
    o = np.argsort(-z)[:5]
    # BH
    m = len(p); orden_p = np.argsort(p)
    q_ = np.empty(m); q_[orden_p] = np.minimum.accumulate(
        (p[orden_p] * m / np.arange(1, m + 1))[::-1])[::-1]
    top_marc[c] = [(NOMBRES[gsel[i]], z[i], auc[i], lfc[i], q_[i]) for i in o]
    say(f"[wilcoxon] c{c} ({anot[c]}): " + "; ".join(
        f"{g} z={zz:.1f} AUC={aa:.3f} lfc={ll:.2f} q={qq:.1e}"
        for g, zz, aa, ll, qq in top_marc[c]))
    say(f"[wilcoxon] c{c}: genes con q<0.05 y lfc>1: "
        f"{int(np.sum((q_<0.05)&(lfc>1)))}")

# ejemplo pequeño de Wilcoxon a mano
xa = np.array([0.0, 1.9, 2.3, 2.8]); xb = np.array([0.0, 0.0, 0.7, 1.1, 0.0])
Rr = stats.rankdata(np.r_[xa, xb])
say(f"[wilcoxon-ej] rangos={Rr.tolist()} R1={Rr[:4].sum()} "
    f"U={Rr[:4].sum()-4*5/2} AUC={(Rr[:4].sum()-10)/20:.3f} "
    f"scipy={stats.mannwhitneyu(xa, xb).statistic}")

# dotplot
genes_dp = ["IL7R", "CCR7", "CD8A", "GZMK", "NKG7", "GNLY", "MS4A1",
            "CD79A", "CD14", "S100A8", "FCGR3A", "MS4A7", "FCER1A", "CST3"]
orden_cl = sorted(range(K), key=lambda c: (TIPOS.index(anot[c]), c))
frac_e = np.array([[np.mean(X[lab == c, IM[g]] > 0) for g in genes_dp]
                   for c in orden_cl])
mean_e = np.array([[Y[lab == c, IM[g]].mean() for g in genes_dp]
                   for c in orden_cl])
mean_s = (mean_e - mean_e.min(0)) / (mean_e.max(0) - mean_e.min(0))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list(
    "curso", ["#F0EFEC", "#CDE2FB", "#86B6EF", "#2A78D6", "#1C5CAB", "#0D366B"])
fig, ax = plt.subplots(figsize=(12.2 * CM, 0.52 * K * CM + 2.4 * CM))
yy, xx_ = np.meshgrid(np.arange(K), np.arange(len(genes_dp)), indexing="ij")
sc = ax.scatter(xx_.ravel(), yy.ravel(), s=frac_e.ravel() * 70,
                c=mean_s.ravel(), cmap=cmap, vmin=0, vmax=1,
                edgecolors=C["base"], linewidths=0.3)
ax.set_xticks(range(len(genes_dp)))
ax.set_xticklabels(genes_dp, rotation=55, ha="right", fontsize=7.5,
                   color=C["tinta2"], style="italic")
ax.set_yticks(range(K))
ax.set_yticklabels([f"{c} · {anot[c]}" for c in orden_cl], fontsize=7.5,
                   color=C["tinta2"])
ax.set_xlim(-0.6, len(genes_dp) - 0.4); ax.set_ylim(K - 0.5, -0.5)
ax.grid(color=C["rejilla"], lw=0.4); ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("expresión media (escalada)", fontsize=7, color=C["tinta2"])
cb.ax.tick_params(labelsize=6.5); cb.outline.set_visible(False)
cb.ax.yaxis.set_major_formatter(coma)
for f_ in (0.25, 0.5, 1.0):
    ax.scatter([], [], s=f_ * 70, c="white", edgecolors=C["gris"],
               linewidths=0.5, label=f"{int(f_*100)} %")
ax.legend(title="% células", loc="upper left", bbox_to_anchor=(1.13, 1.0),
          fontsize=7, title_fontsize=7, labelspacing=1.0, borderpad=0.2)
guardar(fig, "dotplot.pdf")

# ---- efecto de lote y corrección tipo Harmony
say("--- lotes ---")
rngb = np.random.default_rng(31)
fcl = np.exp(rngb.normal(0, 0.45, NG))
Xb1, Tb1, _ = sim_pbmc(1500, seed=101, n_mueren=0, n_dob=0)
Xb2, Tb2, _ = sim_pbmc(1500, seed=202, n_mueren=0, n_dob=0, fc_lote=fcl)
XB = np.vstack([Xb1, Xb2]); TB = np.r_[Tb1, Tb2]
BB = np.r_[np.zeros(len(Xb1), int), np.ones(len(Xb2), int)]
YB = lognorm(XB)
hvB, _, _ = hvg(YB, 500)
pB, _, _ = pca(YB, hvB, 30)


def mezcla(Zx, b, k=30):
    idx = NearestNeighbors(n_neighbors=k + 1).fit(Zx).kneighbors(Zx)[1][:, 1:]
    return np.mean(b[idx] != b[:, None])


def harmony_simple(Z, b, K=20, theta=2.0, sigma=0.1, it=10, lam=1.0,
                   seed=0):
    """Versión didáctica de Harmony: k-medias difuso con penalización de
    diversidad + corrección lineal por mezcla de expertos."""
    rng = np.random.default_rng(seed)
    nb_ = b.max() + 1
    Phi = np.eye(nb_)[b]                      # n x B
    Pr = Phi.mean(0)
    Zc = Z.copy()
    for _ in range(it):
        Zn = Zc / np.linalg.norm(Zc, axis=1, keepdims=True)
        Yk = Zn[rng.choice(len(Zn), K, replace=False)]
        for _ in range(15):
            d = 2 * (1 - Zn @ Yk.T)           # distancia coseno
            Rk = np.exp(-d / sigma)
            Rk /= Rk.sum(1, keepdims=True)
            O = Rk.T @ Phi                     # K x B observado
            Ex = Rk.sum(0)[:, None] * Pr[None]  # K x B esperado
            pen = ((Ex + 1) / (O + 1)) ** theta
            Rk = Rk * (pen @ Phi.T).T
            Rk /= Rk.sum(1, keepdims=True)
            Yk = Rk.T @ Zn
            Yk /= np.linalg.norm(Yk, axis=1, keepdims=True)
        Phi1 = np.c_[np.ones(len(Z)), Phi]
        corr = np.zeros_like(Z)
        for k in range(K):
            Wk = Rk[:, k]
            A_ = Phi1.T @ (Phi1 * Wk[:, None]) + lam * np.diag(
                np.r_[0, np.ones(nb_)])
            beta = np.linalg.solve(A_, Phi1.T @ (Z * Wk[:, None]))
            beta[0] = 0
            corr += Wk[:, None] * (Phi1 @ beta)
        Zc = Z - corr
    return Zc


pBh = harmony_simple(pB, BB)
mz0, mz1 = mezcla(pB, BB), mezcla(pBh, BB)
AB0, _, _ = knn_grafo(pB, 15)
AB1, _, _ = knn_grafo(pBh, 15)
l0 = louvain(AB0, 0.5)[0]; l1 = louvain(AB1, 0.5)[0]
say(f"[lotes] fraccion de vecinos del otro lote (ideal 0.5): antes={mz0:.3f}"
    f" despues={mz1:.3f}")
say(f"[lotes] ARI tipos antes={adjusted_rand_score(TB, l0):.3f} "
    f"despues={adjusted_rand_score(TB, l1):.3f}; "
    f"ARI lote antes={adjusted_rand_score(BB, l0):.3f} "
    f"despues={adjusted_rand_score(BB, l1):.3f}")
UB0 = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(pB)
UB1 = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(pBh)
fig, axs = plt.subplots(2, 2, figsize=(11 * CM, 9.4 * CM))
colL = np.array([C["naranja"], C["azul"]])
for j, (UU, tt) in enumerate(((UB0, "sin corregir"), (UB1, "corregido"))):
    o = np.random.default_rng(1).permutation(len(BB))
    axs[0, j].scatter(UU[o, 0], UU[o, 1], s=1, lw=0, c=colL[BB[o]])
    panel_tipos(axs[1, j], UU, TB, s=1)
    limpio(axs[0, j]); limpio(axs[1, j])
    axs[0, j].set_title(tt)
    for i_, le in ((0, "ab"[j]), (1, "cd"[j])):
        letra(axs[i_, j], le, x=-0.04)
axs[0, 0].scatter([], [], s=12, c=C["naranja"], label="lote 1")
axs[0, 0].scatter([], [], s=12, c=C["azul"], label="lote 2")
axs[0, 0].legend(loc="lower left", fontsize=7, handletextpad=0.1,
                 bbox_to_anchor=(-0.05, -0.08))
for i, nmb in enumerate(TIPOS):
    axs[1, 1].scatter([], [], s=12, c=PAL[i], label=nmb)
fig.legend(*axs[1, 1].get_legend_handles_labels(), loc="lower center",
           ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.07),
           handletextpad=0.1, columnspacing=0.8)
fig.tight_layout(h_pad=1.0)
guardar(fig, "lotes.pdf")

# =====================================================================
# 12.4  Trayectorias: difusión, DPT, PAGA, velocidad de ARN
# =====================================================================
say("=== 12.4 ===")
rngt = np.random.default_rng(77)
NT, GT = 2000, 600
tv = rngt.uniform(0, 1, NT) ** 0.9
rama = np.where(tv < 0.3, 0, rngt.choice([1, 2], NT))
nom_t = (["CD34", "GATA2", "MEIS1", "HLF", "KIT", "GATA1", "KLF1", "TFRC",
          "HBB", "CA1", "CEBPA", "MPO", "ELANE", "LYZ", "CSF1R"]
         + [f"G{i:04d}" for i in range(GT - 15)])
base_t = rngt.lognormal(-0.5, 1.2, GT)
base_t[:15] = rngt.lognormal(-1.0, 0.2, 15)
lfT = np.zeros((NT, GT))


def sig(x, c, s=0.08):
    return 1 / (1 + np.exp(-(x - c) / s))


prog = [0, 1, 2, 3, 4] + list(range(15, 55))
eri = [5, 6, 7, 8, 9] + list(range(55, 95))
mie = [10, 11, 12, 13, 14] + list(range(95, 135))
trans = list(range(135, 160))
amp = rngt.uniform(2.0, 4.0, GT)
cen = rngt.uniform(0.45, 0.85, GT)
cen[[5, 10]] = 0.38; cen[[6, 11]] = 0.5; cen[[8, 12]] = 0.8
cen[[9, 13]] = 0.9; cen[[7, 14]] = 0.65
for g in prog:
    lfT[:, g] = amp[g] * (1 - sig(tv, rngt.uniform(0.2, 0.5), 0.12))
for g in eri:
    lfT[:, g] = amp[g] * sig(tv, cen[g]) * (rama == 1)
for g in mie:
    lfT[:, g] = amp[g] * sig(tv, cen[g]) * (rama == 2)
for g in trans:
    lfT[:, g] = amp[g] * np.exp(-((tv - 0.32) / 0.08) ** 2)
PT = base_t[None] * np.exp(lfT)
PT /= PT.sum(1, keepdims=True)
LT = rngt.lognormal(math.log(3000), 0.35, NT)
thT = rngt.lognormal(math.log(8), 0.4, GT)
XT = nb(rngt, LT[:, None] * PT, thT[None])
YT = lognorm(XT)
ZT = YT - YT.mean(0)
ZT /= ZT.std(0) + 1e-9
Ut, St, Vtt = np.linalg.svd(ZT, full_matrices=False)
pT = Ut[:, :20] * St[:20]

# mapa de difusión (núcleo gaussiano con sigma local, Haghverdi 2016)
kT = 30
nnT = NearestNeighbors(n_neighbors=kT).fit(pT)
dT, iT = nnT.kneighbors(pT)
sig_loc = dT[:, kT // 2]
rows = np.repeat(np.arange(NT), kT)
cols = iT.ravel()
dd = dT.ravel() ** 2
si, sj = sig_loc[rows], sig_loc[cols]
kv = np.sqrt(2 * si * sj / (si ** 2 + sj ** 2)) * np.exp(-dd / (si ** 2 + sj ** 2))
Kmat = sparse.csr_matrix((kv, (rows, cols)), shape=(NT, NT))
Kmat = Kmat.maximum(Kmat.T)
qd = np.asarray(Kmat.sum(1)).ravel()
Kt = sparse.diags(1 / qd) @ Kmat @ sparse.diags(1 / qd)   # alfa=1
dt_ = np.asarray(Kt.sum(1)).ravel()
Msym = sparse.diags(dt_ ** -0.5) @ Kt @ sparse.diags(dt_ ** -0.5)
lam_, V_ = eigsh(Msym, k=12, which="LA")
o = np.argsort(-lam_)
lam_, V_ = lam_[o], V_[:, o]
psi = V_ * (dt_ ** -0.5)[:, None]
psi /= np.linalg.norm(psi, axis=0)
say(f"[difusion] autovalores={np.round(lam_[:6],4).tolist()}")
nDC = 10
Wd = lam_[1:nDC + 1] / (1 - lam_[1:nDC + 1])
iCD34 = nom_t.index("CD34")
score_prog = YT[:, [nom_t.index(g) for g in ("CD34", "GATA2", "MEIS1", "HLF", "KIT")]].mean(1)
# raíz: extremo del DC más correlacionado con el programa progenitor
cands = np.argsort(-score_prog)[:20]
raiz = cands[np.argmax(np.abs(psi[cands, 1] - np.median(psi[:, 1])))]
DPT = np.linalg.norm((psi[:, 1:nDC + 1] - psi[raiz, 1:nDC + 1]) * Wd, axis=1)
DPT = (DPT - DPT.min()) / (DPT.max() - DPT.min())
rs = stats.spearmanr(DPT, tv).statistic
say(f"[dpt] raiz={raiz} (t real={tv[raiz]:.3f}) Spearman(DPT,t)={rs:.3f}")
for rr in (0, 1, 2):
    say(f"[dpt] rama {rr}: n={np.sum(rama==rr)} Spearman="
        f"{stats.spearmanr(DPT[rama==rr], tv[rama==rr]).statistic:.3f}")
# ejemplo de pesos de difusión
say(f"[dpt] pesos lambda/(1-lambda) DC1..4 = {np.round(Wd[:4],2).tolist()}")

# clusters y PAGA
AT, _, _ = knn_grafo(pT, 15)
labT, _, QT, _ = louvain(AT, 0.6, seed=1)
KT = labT.max() + 1
nT = np.bincount(labT)
Ac = AT.tocoo()
Eo = np.zeros((KT, KT))
np.add.at(Eo, (labT[Ac.row], labT[Ac.col]), 1)
Eo = (Eo + Eo.T) / 2
kdeg = Eo.sum(1)
mtot = Eo.sum() / 2
Eexp = np.outer(kdeg, kdeg) / (2 * mtot)
conn = np.clip(Eo / Eexp, 0, None)
np.fill_diagonal(conn, 0)
conn = np.minimum(conn, 1.0)
say(f"[paga] {KT} clusters; tamanos={nT.tolist()}")
dpt_cl = np.array([np.median(DPT[labT == c]) for c in range(KT)])
rama_cl = [np.bincount(rama[labT == c], minlength=3).argmax() for c in range(KT)]
aristas = [(i, j, conn[i, j]) for i in range(KT) for j in range(i + 1, KT)
           if conn[i, j] > 0.05]
say(f"[paga] aristas con conectividad>0.05: {len(aristas)}; "
    + "; ".join(f"{i}-{j}:{c:.2f}" for i, j, c in aristas))

# figura trayectoria (DC1/DC2)
fig, axs = plt.subplots(1, 3, figsize=(13 * CM, 4.7 * CM),
                        gridspec_kw=dict(width_ratios=[1, 1.18, 1]))
D1, D2 = psi[:, 1], psi[:, 2]
colR = np.array([C["gris"], C["rojo"], C["azul"]])
o = np.random.default_rng(2).permutation(NT)
axs[0].scatter(D1[o], D2[o], s=1.5, lw=0, c=colR[rama[o]])
for rr, nmb in enumerate(("tronco", "eritroide", "mieloide")):
    axs[0].scatter([], [], s=12, c=colR[rr], label=nmb)
axs[0].legend(fontsize=6.5, loc="best", handletextpad=0.1)
axs[0].set_title("rama simulada")
cmap_pt = LinearSegmentedColormap.from_list(
    "pt", ["#0D366B", "#2A78D6", "#1BAF7A", "#EDA100", "#EB6834"])
s_ = axs[1].scatter(D1[o], D2[o], s=1.5, lw=0, c=DPT[o], cmap=cmap_pt)
axs[1].scatter(D1[raiz], D2[raiz], s=45, marker="*", c=C["tinta"],
               edgecolors="white", linewidths=0.5, zorder=3)
axs[1].annotate("raíz", (D1[raiz], D2[raiz]), xytext=(-6, 5), ha="right",
                textcoords="offset points", fontsize=7, color=C["tinta"])
cb = fig.colorbar(s_, ax=axs[1], fraction=0.05, pad=0.02)
cb.ax.tick_params(labelsize=6.5); cb.outline.set_visible(False)
cb.ax.yaxis.set_major_formatter(coma)
axs[1].set_title("pseudotiempo (DPT)")
for ax in axs[:2]:
    limpio(ax)
    ax.set_xlabel("DC1", fontsize=7); ax.set_ylabel("DC2", fontsize=7)
axs[2].scatter(tv[o], DPT[o], s=1, lw=0, c=colR[rama[o]], alpha=0.6)
axs[2].set_xlabel("tiempo simulado"); axs[2].set_ylabel("DPT")
axs[2].set_title(f"$\\rho$ = {rs:.3f}".replace(".", ","))
fmt(axs[2])
for ax, le in zip(axs, "abc"):
    letra(ax, le, x=-0.12 if ax is not axs[2] else -0.3, y=1.08)
fig.tight_layout(w_pad=0.8)
guardar(fig, "trayectoria.pdf")

# PAGA + tendencias de genes
fig, axs = plt.subplots(1, 2, figsize=(13 * CM, 5.4 * CM),
                        gridspec_kw=dict(width_ratios=[1, 1.25]))
ax = axs[0]
pos = np.array([[np.median(D1[labT == c]), np.median(D2[labT == c])]
                for c in range(KT)])
ax.scatter(D1, D2, s=0.8, lw=0, c=C["rejilla"], zorder=0)
for i, j, cc in aristas:
    ax.plot(pos[[i, j], 0], pos[[i, j], 1], color=C["tinta2"],
            lw=0.5 + 4 * cc, alpha=0.35 + 0.6 * cc, zorder=1,
            solid_capstyle="round")
sc2 = ax.scatter(pos[:, 0], pos[:, 1], s=30 + nT / 2.5, c=dpt_cl,
                 cmap=cmap_pt, vmin=0, vmax=1, edgecolors="white",
                 linewidths=0.8, zorder=2)
for c in range(KT):
    ax.text(pos[c, 0], pos[c, 1], str(c), fontsize=6.5, color="white",
            ha="center", va="center", fontweight="bold", zorder=3)
limpio(ax); ax.set_title("grafo PAGA")
ax.set_xlabel("DC1", fontsize=7); ax.set_ylabel("DC2", fontsize=7)
letra(ax, "a", x=-0.05)
ax = axs[1]
for g, cc, rr in (("CD34", C["gris"], None), ("GATA1", C["rojo"], 1),
                  ("HBB", C["rojooscuro"], 1), ("MPO", C["azul"], 2),
                  ("LYZ", C["azulnoche"], 2)):
    gi = nom_t.index(g)
    for rsel, ls in ((1, "-"), (2, "--")) if rr is None else ((rr, "-"),):
        m = (rama == 0) | (rama == rsel)
        xb = np.linspace(0, 1, 26)
        dig = np.clip(np.digitize(DPT[m], xb) - 1, 0, 24)
        yv = np.array([YT[m, gi][dig == b].mean() if np.any(dig == b)
                       else np.nan for b in range(25)])
        xm = (xb[:-1] + xb[1:]) / 2
        ok_ = np.isfinite(yv)
        ax.plot(xm[ok_], yv[ok_], color=cc, lw=1.3, ls=ls,
                label=g if (rr is not None or rsel == 1) else None)
ax.set_xlabel("pseudotiempo (DPT)"); ax.set_ylabel("log(1 + CP10k)")
ax.legend(fontsize=6.5, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.0),
          handlelength=1.6)
ax.set_title("expresión a lo largo del pseudotiempo")
fmt(ax)
letra(ax, "b", x=-0.2)
fig.tight_layout(w_pad=0.6)
guardar(fig, "paga_genes.pdf")

# velocidad de ARN: un gen, modelo cinético
al, be, ga = 4.0, 1.0, 0.4
ts = 6.0


def uv(t, u0, s0, a):
    u = u0 * np.exp(-be * t) + a / be * (1 - np.exp(-be * t))
    s = (s0 * np.exp(-ga * t) + a / ga * (1 - np.exp(-ga * t))
         + (a - be * u0) / (ga - be) * (np.exp(-ga * t) - np.exp(-be * t)))
    return u, s


rgv = np.random.default_rng(12)
n_in, n_rep = 260, 200
t_in = rgv.uniform(0, ts, n_in)
u_in, s_in = uv(t_in, 0, 0, al)
us, ss = uv(np.array([ts]), 0, 0, al)
t_re = rgv.uniform(0, 8, n_rep)
u_re, s_re = uv(t_re, us[0], ss[0], 0.0)
U_ = np.r_[u_in, u_re]; S_ = np.r_[s_in, s_re]
fase = np.r_[np.ones(n_in, int), np.zeros(n_rep, int)]
Uo = rgv.poisson(U_ * 3) / 3
So = rgv.poisson(S_ * 3) / 3
qq = (Uo >= np.quantile(Uo, 0.95)) | (So >= np.quantile(So, 0.95))
qq |= (Uo <= np.quantile(Uo, 0.05)) & (So <= np.quantile(So, 0.05))
g_hat = np.sum(Uo[qq] * So[qq]) / np.sum(So[qq] ** 2)
vel = Uo - g_hat * So
acc = np.mean((vel > 0) == (fase == 1))
say(f"[velocidad] alfa={al} beta={be} gamma={ga}; gamma/beta real={ga/be:.3f}"
    f" estimado (extremos)={g_hat:.3f}; signo correcto={acc*100:.1f}%")
say(f"[velocidad] estado estacionario u*={al/be:.2f} s*={al/ga:.2f}")
w("fase.dat", "s u fase\n" + "\n".join(
    f"{s:.4f} {u:.4f} {f}" for s, u, f in zip(So, Uo, fase)))
tt = np.linspace(0, ts, 80)
uu, sv = uv(tt, 0, 0, al)
tt2 = np.linspace(0, 12, 80)
uu2, sv2 = uv(tt2, us[0], ss[0], 0.0)
w("fase_curva.dat", "s u\n" + "\n".join(
    f"{s:.4f} {u:.4f}" for s, u in zip(np.r_[sv, sv2], np.r_[uu, uu2])))
w("fase_param.tex",
  f"\\def\\capdocegamma{{{g_hat:.4f}}}\\def\\capdocegreal{{{ga/be:.4f}}}\n")

w("cifras.txt", "\n".join(CIFRAS) + "\n")
print("OK")
