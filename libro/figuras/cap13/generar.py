"""Genera datos y figuras del capítulo 13 (epigenómica y regulación).

Ejecutar desde libro/:  python3 figuras/cap13/generar.py
Escribe .dat/.tex en figuras/cap13/ e imprime (y guarda en cifras.txt)
las cifras que se citan en los ejemplos del texto (única fuente de verdad).
Todos los datos son SIMULADOS con modelos explícitos y semilla fija.
"""
from pathlib import Path
import math
import numpy as np
from scipy import stats

OUT = Path(__file__).parent
rng = np.random.default_rng(1313)
LOG = []


def w(name, text):
    (OUT / name).write_text(text)


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def dat(name, header, rows, fmt="{:.5g}"):
    lines = [header] + [" ".join(fmt.format(v) for v in r) for r in rows]
    w(name, "\n".join(lines) + "\n")


# =====================================================================
# 13.1-a  Perfiles de extremos 5' por hebra alrededor de un sitio
# =====================================================================
rng = np.random.default_rng(101)
L_READ = 36
n_frag = 4000
flen = np.clip(rng.normal(200, 30, n_frag).round().astype(int), 90, 350)
centro = rng.normal(0, 12, n_frag)             # centro del fragmento ~ sitio
izq = np.round(centro - flen / 2).astype(int)   # 5' de la lectura +
der = izq + flen - 1                            # 5' de la lectura -
xs = np.arange(-500, 501)
bw = 15


def dens(v):
    k = stats.gaussian_kde(v, bw_method=bw / v.std())
    return k(xs) * len(v)


dp, dm = dens(izq), dens(der)
d_hat = int(np.median(flen))
comb = dens(np.concatenate([izq + d_hat // 2, der - d_hat // 2]))
# lecturas extendidas a d: cobertura
cov = np.zeros_like(xs, dtype=float)
for a in izq:
    cov[max(a + 500, 0):min(a + d_hat + 500, 1001)] += 1
for b in der:
    cov[max(b - d_hat + 1 + 500, 0):min(b + 1 + 500, 1001)] += 1
sc = 1.0 / n_frag * 100
dat("hebras.dat", "x plus minus comb cov",
    [(x, dp[i] * sc, dm[i] * sc, comb[i] * sc / 2, cov[i] / n_frag * 100)
     for i, x in enumerate(xs) if i % 5 == 0])
mp, mm = xs[np.argmax(dp)], xs[np.argmax(dm)]
say(f"[hebras] moda +: {mp}  moda -: {mm}  distancia: {mm - mp}"
    f"  mediana long. fragmento: {d_hat}")

# =====================================================================
# 13.1-b  Correlación cruzada entre hebras (con pico fantasma)
# =====================================================================
rng = np.random.default_rng(102)
G = 3_000_000
# mapabilidad: bloques únicos / repetidos
mapeable = np.ones(G, bool)
pos = 0
while pos < G:
    u = int(rng.exponential(260))
    r = int(rng.exponential(60))
    mapeable[pos + u:pos + u + r] = False
    pos += u + r
sitios = rng.choice(np.arange(5000, G - 5000), 900, replace=False)
fr_len = np.clip(rng.normal(180, 25, 400_000).round().astype(int), 80, 320)


# densidad regional del fondo (sesgos de GC, número de copias...)
tasa = np.repeat(rng.lognormal(0, 0.6, G // 20000 + 1), 20000)[:G]
tasa /= tasa.sum()


def simula_lecturas(n_senal, n_ruido):
    c = np.concatenate([
        rng.choice(sitios, n_senal) + rng.normal(0, 15, n_senal).round(),
        rng.choice(G, n_ruido, p=tasa)]).astype(int)
    fl = rng.choice(fr_len, c.size)
    a = c - fl // 2
    b = a + fl - 1
    hebra = rng.random(c.size) < 0.5
    p5 = np.where(hebra, a, b)                  # extremo 5' leído
    start = np.where(hebra, a, b - L_READ + 1)  # inicio de la lectura
    ok = (start > 0) & (start < G - L_READ)
    ok[ok] &= mapeable[start[ok]]
    return p5[ok & hebra], p5[ok & ~hebra]


def xcor(pl, mi, shifts):
    cp = np.bincount(pl, minlength=G + 400)[:G].astype(float)
    cm = np.bincount(mi, minlength=G + 400)[:G].astype(float)
    out = []
    for s in shifts:
        a, b = cp[:G - s], cm[s:]
        out.append(np.corrcoef(a, b)[0, 1])
    return np.array(out)


shifts = np.arange(0, 401, 4)
res = {}
for nombre, ns, nr in [("bueno", 60_000, 540_000), ("pobre", 14_000, 586_000)]:
    pl, mi = simula_lecturas(ns, nr)
    res[nombre] = xcor(pl, mi, shifts)
    cc = res[nombre]
    imax = np.argmax(np.where(shifts > 100, cc, -1))
    iread = np.argmin(abs(shifts - L_READ))
    cmin = cc.min()
    nsc = cc[imax] / cmin
    rsc = (cc[imax] - cmin) / (cc[iread] - cmin)
    say(f"[xcor] {nombre}: frag={shifts[imax]}  cc_frag={cc[imax]:.4f} "
        f" cc_read={cc[iread]:.4f}  cc_min={cmin:.4f}  NSC={nsc:.2f}"
        f"  RSC={rsc:.2f}")
    res[nombre + "_q"] = (shifts[imax], nsc, rsc)
dat("xcor.dat", "s bueno pobre",
    [(s, res["bueno"][i] * 100, res["pobre"][i] * 100)
     for i, s in enumerate(shifts)])

# =====================================================================
# 13.1-c  Pico sobre fondo de Poisson local (pileup y lambda)
# =====================================================================
rng = np.random.default_rng(103)
N_T, N_C, GEF, D = 20e6, 25e6, 2.7e9, 200
W2 = 2 * D
lam_bg = N_T * W2 / GEF
ratio = N_T / N_C
c1k, c5k, c10k = 10, 45, 95
l1 = c1k * W2 / 1000 * ratio
l5 = c5k * W2 / 5000 * ratio
l10 = c10k * W2 / 10000 * ratio
lam_loc = max(lam_bg, l1, l5, l10)
k_obs = 17
p_loc = stats.poisson.sf(k_obs - 1, lam_loc)
p_bg = stats.poisson.sf(k_obs - 1, lam_bg)
say(f"[poisson] lam_BG={lam_bg:.3f} l1k={l1:.3f} l5k={l5:.3f} "
    f"l10k={l10:.3f} lam_local={lam_loc:.3f}")
say(f"[poisson] k={k_obs}: p_local={p_loc:.3e} (-log10={-math.log10(p_loc):.2f})"
    f"  p_BG={p_bg:.3e}")
# región con amplificación en el control: la corrección local salva
c10k_amp = 400
l10a = c10k_amp * W2 / 10000 * ratio
k2 = 14
say(f"[poisson] region amplificada: l10k={l10a:.2f}; k={k2}: "
    f"p_BG={stats.poisson.sf(k2 - 1, lam_bg):.2e}  "
    f"p_local={stats.poisson.sf(k2 - 1, max(lam_bg, l10a)):.3f}")
for k in range(5, 20):
    if stats.poisson.sf(k - 1, lam_loc) < 1e-5:
        say(f"[poisson] umbral 1e-5 con lam_local: k>={k}")
        break
dat("pmf.dat", "k pmf", [(k, stats.poisson.pmf(k, lam_loc))
                         for k in range(0, 21)])
# pileup a lo largo de 10 kb
xr = np.arange(0, 10001, 25)
bg = rng.poisson(lam_bg * 25 / W2 * 8, xr.size).astype(float)
bg = np.convolve(bg, np.ones(8) / 8 * 8, mode="same") / 8 * W2 / 25 / 8
base = lam_bg + 0.6 * np.sin(xr / 1400) + rng.normal(0, 0.35, xr.size)
pile = base + 15 * np.exp(-0.5 * ((xr - 5200) / 160) ** 2) \
    + 3.2 * np.exp(-0.5 * ((xr - 2300) / 220) ** 2)
pile = np.clip(pile, 0.2, None)
ctrl = lam_bg + 0.3 * np.sin(xr / 900 + 1) + 0.25 * rng.normal(size=xr.size)
ctrl = np.clip(ctrl, 0.5, None)
lam_l = np.maximum(lam_bg, np.convolve(ctrl, np.ones(41) / 41, "same"))
lam_l[:20] = lam_l[20]
lam_l[-20:] = lam_l[-21]
dat("pileup.dat", "x pile ctrl laml", [(x / 1000, pile[i], ctrl[i], lam_l[i])
                                       for i, x in enumerate(xr)])

# =====================================================================
# 13.1-d  IDR: rangos de dos réplicas
# =====================================================================
rng = np.random.default_rng(104)
n_true, n_false = 1500, 2500
s_true = rng.normal(0, 1, n_true)
r1t = s_true + rng.normal(0, 0.35, n_true) + 2.6
r2t = s_true + rng.normal(0, 0.35, n_true) + 2.6
r1f = rng.normal(0, 1, n_false)
r2f = rng.normal(0, 1, n_false)
s1 = np.concatenate([r1t, r1f])
s2 = np.concatenate([r2t, r2f])
rk1 = stats.rankdata(-s1)
rk2 = stats.rankdata(-s2)
etiq = np.r_[np.ones(n_true), np.zeros(n_false)]
idx = rng.permutation(s1.size)
dat("idr_true.dat", "r1 r2", [(rk1[i], rk2[i]) for i in idx if etiq[i] == 1])
dat("idr_false.dat", "r1 r2", [(rk1[i], rk2[i]) for i in idx if etiq[i] == 0])
rho = stats.spearmanr(rk1, rk2).correlation
top = (rk1 <= 1000) & (rk2 <= 1000)
say(f"[idr] Spearman global={rho:.2f}; en top1000 de ambas: {top.sum()} "
    f"picos, verdaderos={int(etiq[top].sum())}")

# FRiP
say(f"[frip] 3.1e6/20e6 = {3.1e6 / 20e6:.3f}")

# =====================================================================
# 13.1-e  ATAC-seq: tamaños de fragmento y enriquecimiento en TSS
# =====================================================================
rng = np.random.default_rng(105)
n = 2_000_000
comp = rng.choice(4, n, p=[0.52, 0.30, 0.13, 0.05])
lens = np.empty(n)
m0 = comp == 0
lens[m0] = rng.gamma(4.0, 16.0, m0.sum()) + 20
for c, mu, sd in [(1, 195, 28), (2, 385, 38), (3, 575, 50)]:
    m = comp == c
    lens[m] = rng.normal(mu, sd, m.sum())
# periodicidad helicoidal (10,5 pb): aceptación modulada
acc = 1 + 0.18 * np.cos(2 * np.pi * (lens - 5) / 10.5)
keep = rng.random(n) < acc / 1.18
lens = lens[keep & (lens > 20) & (lens < 1000)].round().astype(int)
h = np.bincount(lens, minlength=1001)[:1000] / lens.size
dat("atac_frag.dat", "l f", [(l, max(h[l], 1e-7) * 1000)
                             for l in range(25, 1000)])
nfr = (lens < 100).mean()
mono = ((lens >= 180) & (lens <= 247)).mean()
say(f"[atac] fragmentos: {lens.size}; <100pb={nfr:.3f}; 180-247={mono:.3f}")
xs_t = np.arange(-2000, 2001, 10)
perfil = 1 + 7.5 * np.exp(-0.5 * (xs_t / 90) ** 2)
perfil += 1.2 * np.exp(-0.5 * ((xs_t - 250) / 45) ** 2) * 0  # sin sesgo
dip = -0.8 * np.exp(-0.5 * ((xs_t - 180) / 50) ** 2)
perfil = perfil + dip + 0.6 * np.exp(-0.5 * ((xs_t - 330) / 55) ** 2)
perfil *= 1 + rng.normal(0, 0.04, xs_t.size)
fl = np.r_[perfil[:10], perfil[-10:]].mean()
perfil /= fl
say(f"[atac] TSS enrichment (max del perfil normalizado): {perfil.max():.2f}")
dat("tss.dat", "x e", [(x, perfil[i]) for i, x in enumerate(xs_t)])

# =====================================================================
# 13.2-a  Beta y M valores
# =====================================================================
rng = np.random.default_rng(106)
ncg = 400_000
grupo = rng.random(ncg)
beta = np.where(grupo < 0.33, rng.beta(0.9, 14, ncg),
                np.where(grupo < 0.43, rng.beta(3, 3, ncg),
                         rng.beta(14, 2.2, ncg)))
Mi = rng.gamma(20, 250, ncg)          # intensidad total
Meth = beta * Mi
Unm = (1 - beta) * Mi
b_obs = Meth / (Meth + Unm + 100)
m_obs = np.log2((Meth + 1) / (Unm + 1))
hb, eb = np.histogram(b_obs, bins=50, range=(0, 1), density=True)
hm, em = np.histogram(m_obs, bins=60, range=(-8, 6), density=True)
dat("beta_hist.dat", "x d", [(eb[i], hb[i]) for i in range(50)] + [(1, 0)])
dat("m_hist.dat", "x d", [(em[i], hm[i]) for i in range(60)] + [(6, 0)])
say(f"[beta] fracción beta<0.2: {(b_obs < 0.2).mean():.3f}; "
    f">0.8: {(b_obs > 0.8).mean():.3f}")
# heterocedasticidad: var de beta vs M en repeticiones técnicas
for b in [0.05, 0.5, 0.95]:
    tot = rng.gamma(20, 250, 2000)
    me = rng.poisson(b * tot)
    un = rng.poisson((1 - b) * tot)
    bb = me / (me + un + 100)
    mm = np.log2((me + 1) / (un + 1))
    say(f"[hetero] beta={b}: sd(beta)={bb.std():.4f} sd(M)={mm.std():.4f}")

# ejemplo: 17 C / 3 T
C, T = 17, 3
bhat = C / (C + T)
say(f"[ej-meth] beta={bhat:.3f}  M(offset1)={math.log2((C + 1) / (T + 1)):.3f}"
    f"  M(sin offset)={math.log2(C / T):.3f}")
se = math.sqrt(bhat * (1 - bhat) / 20)
say(f"[ej-meth] EE binomial={se:.4f}  IC95 Wald=({bhat - 1.96 * se:.3f},"
    f"{bhat + 1.96 * se:.3f})")
lo, hi = stats.beta.ppf([0.025, 0.975], C + 0.5, T + 0.5)
say(f"[ej-meth] IC95 Jeffreys=({lo:.3f},{hi:.3f})")
for phi in [0.0, 0.05, 0.1]:
    v = 20 * bhat * (1 - bhat) * (1 + 19 * phi)
    say(f"[ej-meth] var betabinom n=20 phi={phi}: {v:.3f}  sd={math.sqrt(v):.3f}")

# prueba de Wald tipo DSS en un CpG: 3 vs 3 réplicas
A = [(17, 20), (22, 25), (14, 18)]
B = [(6, 21), (9, 24), (5, 19)]


def grupo_est(g, phi):
    mu = sum(c for c, _ in g) / sum(n for _, n in g)
    var = sum(n * mu * (1 - mu) * (1 + (n - 1) * phi) / n ** 2 for _, n in g)
    var /= len(g) ** 2
    return mu, var


phi = 0.05
ma, va = grupo_est(A, phi)
mb, vb = grupo_est(B, phi)
z = (ma - mb) / math.sqrt(va + vb)
say(f"[wald] muA={ma:.3f} muB={mb:.3f} varA={va:.5f} varB={vb:.5f} "
    f"z={z:.2f} p={2 * stats.norm.sf(abs(z)):.2e}")
ma0, va0 = grupo_est(A, 0.0)
mb0, vb0 = grupo_est(B, 0.0)
z0 = (ma0 - mb0) / math.sqrt(va0 + vb0)
say(f"[wald] sin dispersión (binomial): z={z0:.2f} "
    f"p={2 * stats.norm.sf(abs(z0)):.2e}")
for phi2 in [0.02, 0.1, 0.2]:
    _, va2 = grupo_est(A, phi2)
    _, vb2 = grupo_est(B, phi2)
    z2 = (ma - mb) / math.sqrt(va2 + vb2)
    say(f"[wald] phi={phi2}: z={z2:.2f} p={2 * stats.norm.sf(abs(z2)):.2e}")

# O/E de CpG (Gardiner-Garden & Frommer)
def sintetica(L, gc, keep_cg, semilla):
    g = np.random.default_rng(semilla)
    p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
    s = list("".join(g.choice(list("ACGT"), L, p=p)))
    for i in range(L - 1):   # desaminación de 5mC: CG -> TG / CA
        if s[i] == "C" and s[i + 1] == "G" and g.random() > keep_cg:
            if g.random() < 0.5:
                s[i] = "T"
            else:
                s[i + 1] = "A"
    return "".join(s)


isla = sintetica(100, 0.74, 0.9, 11)
mar = sintetica(100, 0.41, 0.22, 5)
for nom, s in [("isla", isla), ("mar", mar)]:
    nC, nG, nCG = s.count("C"), s.count("G"), s.count("CG")
    L = len(s)
    oe = nCG * L / (nC * nG)
    say(f"[cpg] {nom}: L={L} C={nC} G={nG} CG={nCG} GC={(nC + nG) / L:.3f}"
        f" O/E={oe:.3f}")
say(f"[cpg] isla seq: {isla}")
say(f"[cpg] mar seq: {mar}")

# =====================================================================
# 13.2-b  DMR simulada: 2 grupos x 3 réplicas, beta-binomial
# =====================================================================
rng = np.random.default_rng(107)
pos_cg = np.sort(rng.choice(np.arange(0, 3000), 55, replace=False))
mu_a = 0.82 - 0.05 * np.sin(pos_cg / 500)
mu_b = mu_a.copy()
dmr = (pos_cg > 1100) & (pos_cg < 1900)
mu_b[dmr] = 0.25 + 0.05 * np.cos(pos_cg[dmr] / 200)
phi_s = 0.05
rows = []
mean_a, mean_b, nA, nB = [], [], [], []
for j, p in enumerate(pos_cg):
    ra, rb, sa, sb = [], [], 0, 0
    for g, mu in [("a", mu_a[j]), ("b", mu_b[j])]:
        for r in range(3):
            nn = max(rng.poisson(14), 3)
            a_ = mu * (1 - phi_s) / phi_s
            b_ = (1 - mu) * (1 - phi_s) / phi_s
            pp = rng.beta(a_, b_)
            cc = rng.binomial(nn, pp)
            (ra if g == "a" else rb).append((cc, nn))
    rows.append((p, ra, rb))
pts_a, pts_b, zs = [], [], []
for p, ra, rb in rows:
    for c_, n_ in ra:
        pts_a.append((p, c_ / n_))
    for c_, n_ in rb:
        pts_b.append((p, c_ / n_))
    m1, v1 = grupo_est(ra, phi_s)
    m2, v2 = grupo_est(rb, phi_s)
    zs.append((m1 - m2) / math.sqrt(max(v1 + v2, 1e-6)))
zs = np.array(zs)
pv = 2 * stats.norm.sf(np.abs(zs))
dat("dmr_a.dat", "x b", pts_a)
dat("dmr_b.dat", "x b", pts_b)
# suavizado (media ponderada gaussiana, 150 pb)
grid = np.arange(0, 3001, 25)


def suave(pts):
    P = np.array(pts)
    out = []
    for g in grid:
        wts = np.exp(-0.5 * ((P[:, 0] - g) / 150) ** 2)
        out.append((wts * P[:, 1]).sum() / wts.sum())
    return np.array(out)


sa_, sb_ = suave(pts_a), suave(pts_b)
dat("dmr_curva.dat", "x a b", [(g, sa_[i], sb_[i]) for i, g in enumerate(grid)])
sig = pv < 0.01
say(f"[dmr] CpG totales={len(pos_cg)} en DMR real={dmr.sum()} "
    f"significativos p<0.01={sig.sum()} (dentro={int((sig & dmr).sum())})")
dat("dmr_p.dat", "x lp", [(p, -math.log10(max(pv[j], 1e-12)))
                          for j, p in enumerate(pos_cg)])
# límites de la DMR llamada: tramo contiguo más largo de significativos
best, cur, st = (0, 0, 0), 0, 0
for j in range(len(sig)):
    if sig[j]:
        if cur == 0:
            st = j
        cur += 1
        if cur > best[0]:
            best = (cur, st, j)
    else:
        cur = 0
say(f"[dmr] DMR llamada: CpG {best[1]}..{best[2]} "
    f"pos {pos_cg[best[1]]}-{pos_cg[best[2]]} ({best[0]} CpG)")
w("dmr_lim.tex", f"\\def\\capdmrini{{{pos_cg[best[1]]}}}"
  f"\\def\\capdmrfin{{{pos_cg[best[2]]}}}")

# =====================================================================
# 13.3  Descubrimiento de motivos: EM (OOPS) y muestreo de Gibbs
# =====================================================================
rng = np.random.default_rng(108)
BASES = "ACGT"
TRUE = np.array([  # columnas A C G T; consenso ATGASTCA (tipo AP-1)
    [0.60, 0.15, 0.15, 0.10],
    [0.05, 0.05, 0.05, 0.85],
    [0.05, 0.05, 0.85, 0.05],
    [0.88, 0.04, 0.04, 0.04],
    [0.05, 0.47, 0.43, 0.05],
    [0.05, 0.05, 0.05, 0.85],
    [0.05, 0.85, 0.05, 0.05],
    [0.88, 0.04, 0.04, 0.04],
])
Wm = TRUE.shape[0]
NSEQ, LSEQ = 30, 80
seqs = rng.integers(0, 4, (NSEQ, LSEQ))
sitio_real = rng.integers(0, LSEQ - Wm + 1, NSEQ)
for i in range(NSEQ):
    for k in range(Wm):
        seqs[i, sitio_real[i] + k] = rng.choice(4, p=TRUE[k])
p0 = np.full(4, 0.25)
M = LSEQ - Wm + 1
# ventanas: (NSEQ, M, Wm)
win = np.stack([seqs[:, j:j + Wm] for j in range(M)], axis=1)


def ll_ratio(theta):
    lr = np.log2(theta[np.arange(Wm), win]).sum(-1) - Wm * np.log2(0.25)
    mx = lr.max(1, keepdims=True)
    return float((np.log2(np.exp2(lr - mx).mean(1)) + mx[:, 0]).sum()), lr


def em(theta, iters=30, beta=0.1):
    tray = []
    for _ in range(iters):
        ll, lr = ll_ratio(theta)
        tray.append(ll)
        Z = np.exp2(lr - lr.max(1, keepdims=True))
        Z /= Z.sum(1, keepdims=True)
        cnt = np.zeros((Wm, 4))
        for b in range(4):
            cnt[:, b] = (Z[:, :, None] * (win == b)).sum((0, 1))
        theta = (cnt + beta) / (cnt + beta).sum(1, keepdims=True)
    tray.append(ll_ratio(theta)[0])
    return theta, tray, Z


def semilla(kmer, fuerza=0.5):
    th = np.full((Wm, 4), (1 - fuerza) / 3)
    th[np.arange(Wm), kmer] = fuerza
    return th


def ic(theta):
    return float((theta * np.log2(theta / 0.25)).sum())


runs = []
starts = [(3, 11), (17, 40), (8, 2), (22, 60), (sitio_real[5] and 5, None)]
rng_em = np.random.default_rng(7)
em_res = []
for r in range(6):
    if r == 5:   # semilla desde un sitio real (como hace MEME al barrer)
        i0 = 12
        km = seqs[i0, sitio_real[i0]:sitio_real[i0] + Wm]
    else:
        i0 = rng_em.integers(NSEQ)
        j0 = rng_em.integers(M)
        km = seqs[i0, j0:j0 + Wm]
    th0 = semilla(km)
    th, tray, Z = em(th0)
    em_res.append((th0, th, tray, Z, "".join(BASES[b] for b in km)))
    say(f"[em] corrida {r}: semilla={em_res[-1][4]} LLR0={tray[0]:.1f} "
        f"LLRfin={tray[-1]:.1f} IC={ic(th):.2f} "
        f"consenso={''.join(BASES[b] for b in th.argmax(1))}")
dat("em_ll.dat", "it " + " ".join(f"r{r}" for r in range(6)),
    [(t, *[er[2][t] for er in em_res]) for t in range(31)])
best_r = int(np.argmax([er[2][-1] for er in em_res[:5]]))  # semilla aleatoria
th_best, Zb = em_res[best_r][1], em_res[best_r][3]
acierto = int((np.abs(Zb.argmax(1) - sitio_real) == 0).sum())
say(f"[em] mejor corrida={best_r}; sitios recuperados exactos: {acierto}/{NSEQ}")
th_it1 = em(em_res[best_r][0], iters=6)[0]
th_it3 = em(em_res[best_r][0], iters=8)[0]
say(f"[em] IC semilla={ic(em_res[best_r][0]):.2f} it6={ic(th_it1):.2f} "
    f"it8={ic(th_it3):.2f} final={ic(th_best):.2f} verdadero={ic(TRUE):.2f}")
# E-step de ejemplo: razón para el sitio real en la secuencia 0 con TRUE
lr_all = np.log2(TRUE[np.arange(Wm), win]).sum(-1) + 2 * Wm
i_ej = int(np.argmax(lr_all[np.arange(NSEQ), sitio_real]))
lr0 = lr_all[i_ej]
Z0 = np.exp2(lr0 - lr0.max())
Z0 /= Z0.sum()
sr = sitio_real[i_ej]
say(f"[estep] seq{i_ej} sitio real={sr} "
    f"kmer={''.join(BASES[b] for b in win[i_ej, sr])} "
    f"LLR={lr0[sr]:.2f} bits  Z={Z0[sr]:.4f}  "
    f"segundo mayor Z={np.sort(Z0)[-2]:.2e}  suma otros={1 - Z0[sr]:.2e}")
say(f"[estep] mediana LLR sitios reales="
    f"{np.median(lr_all[np.arange(NSEQ), sitio_real]):.2f}")


# ---- Gibbs (site sampler de Lawrence et al.) ----
def gibbs(seed, sweeps=60, beta=0.25):
    g = np.random.default_rng(seed)
    a = g.integers(0, M, NSEQ)
    tray = []
    for sw in range(sweeps):
        for z in g.permutation(NSEQ):
            cnt = np.full((Wm, 4), beta)
            for i in range(NSEQ):
                if i != z:
                    cnt[np.arange(Wm), win[i, a[i]]] += 1
            q = cnt / cnt.sum(1, keepdims=True)
            lr = np.log2(q[np.arange(Wm), win[z]]).sum(-1) + 2 * Wm
            pr = np.exp2(lr - lr.max())
            a[z] = g.choice(M, p=pr / pr.sum())
        cnt = np.full((Wm, 4), beta)
        for i in range(NSEQ):
            cnt[np.arange(Wm), win[i, a[i]]] += 1
        q = cnt / cnt.sum(1, keepdims=True)
        tray.append(ll_ratio(q)[0])
    return q, tray, a


gb = []
for sd in [1, 2, 3, 4]:
    q, tray, a = gibbs(sd)
    exact = int((a == sitio_real).sum())
    gb.append((q, tray, a))
    say(f"[gibbs] cadena {sd}: LLRfin={tray[-1]:.1f} exactos={exact}/{NSEQ} "
        f"desfase +-1..2={int((np.abs(a - sitio_real) <= 2).sum())} "
        f"consenso={''.join(BASES[b] for b in q.argmax(1))} "
        f"primer sweep >= 0.95*max: "
        f"{next(i for i, v in enumerate(tray) if v >= 0.95 * max(tray))}")
dat("gibbs_ll.dat", "it " + " ".join(f"c{c}" for c in range(4)),
    [(t + 1, *[g_[1][t] for g_ in gb]) for t in range(60)])
say(f"[true] LLR con matriz verdadera: {ll_ratio(TRUE)[0]:.1f}; "
    f"IC verdadero={ic(TRUE):.2f}")


# ---- logos TikZ ----
def logo(theta, nombre, xs=0.42, ys=0.62, titulo=""):
    t = [rf"\begin{{tikzpicture}}[x={xs}cm,y={ys}cm]"]
    t.append(rf"\draw[base] (0.4,0) -- ({Wm + 0.6:.2f},0);")
    t.append(r"\draw[base] (0.4,0) -- (0.4,2);")
    for v in (0, 1, 2):
        t.append(rf"\draw[base] (0.4,{v}) -- (0.3,{v}); \node[font=\sffamily"
                 rf"\tiny,text=gris,anchor=east] at (0.3,{v}) {{{v}}};")
    for c in range(Wm):
        f = theta[c]
        R = 2 + (f * np.log2(f)).sum()
        hs = f * R
        y = 0.0
        for bi in np.argsort(hs):
            hh = hs[bi]
            if hh > 0.01:
                t.append(rf"\node[anchor=south west,inner sep=0pt,"
                         rf"text=nuc{BASES[bi]}] at ({c + 0.56:.3f},{y:.4f}) "
                         rf"{{\resizebox{{{0.88 * xs:.3f}cm}}{{{hh * ys:.4f}cm}}"
                         rf"{{\sffamily\bfseries {BASES[bi]}}}}};")
            y += hh
        t.append(rf"\node[font=\sffamily\tiny,text=gris] at ({c + 1},-0.2)"
                 rf" {{{c + 1}}};")
    if titulo:
        t.append(rf"\node[font=\sffamily\scriptsize\bfseries,text=tinta,"
                 rf"anchor=south] at ({(Wm + 1) / 2:.2f},2.05) {{{titulo}}};")
    t.append(r"\end{tikzpicture}")
    w(nombre, "\n".join(t))


logo(TRUE, "logo_real.tex", titulo="motivo implantado")
logo(em_res[best_r][0], "logo_it0.tex", titulo="semilla (iteración 0)")
logo(th_it1, "logo_it1.tex", titulo="EM, iteración 6")
logo(th_it3, "logo_it3.tex", titulo="EM, iteración 8")
logo(th_best, "logo_em.tex", titulo="EM, convergido")
gbest = int(np.argmax([g_[1][-1] for g_ in gb]))
logo(gb[gbest][0], "logo_gibbs.tex", titulo="Gibbs, barrido 60")

# ---- centralidad (CentriMo) ----
NP, LP = 3000, 500
pos_dir = np.clip(rng.normal(0, 22, NP).round(), -240, 240)
tiene = rng.random(NP) < 0.55
pos_dir = pos_dir[tiene]
pos_cof = np.clip(rng.normal(0, 110, int(0.35 * NP)).round(), -240, 240)
pos_fondo = rng.integers(-240, 241, int(0.25 * NP))
bins = np.arange(-250, 251, 20)
hd, _ = np.histogram(pos_dir, bins)
hc, _ = np.histogram(pos_cof, bins)
hf, _ = np.histogram(pos_fondo, bins)
dat("centrimo.dat", "x dir cof fondo",
    [((bins[i] + bins[i + 1]) / 2, hd[i], hc[i], hf[i])
     for i in range(len(hd))])
# prueba binomial de centralidad: sitios en la ventana central de 100 pb
for nom, pp in [("directo", pos_dir), ("cofactor", pos_cof),
                ("fondo", pos_fondo)]:
    k = int((np.abs(pp) <= 50).sum())
    nn = len(pp)
    frac = 101 / 481
    pvb = stats.binom.sf(k - 1, nn, frac)
    say(f"[centrimo] {nom}: n={nn} en +-50={k} ({k / nn:.3f}) "
        f"esperado={frac:.3f} p={pvb:.2e}")

# ---- enriquecimiento diferencial tipo HOMER ----
nt, kt, nb, kb = 2000, 640, 20000, 1200
N_, K_ = nt + nb, kt + kb
p_hg = stats.hypergeom.sf(kt - 1, N_, K_, nt)
say(f"[homer] objetivo {kt}/{nt}={kt / nt:.3f} fondo {kb}/{nb}={kb / nb:.3f}"
    f" p_hiper={p_hg:.2e} ln p={math.log(p_hg):.1f}"
    f" enriquecimiento={(kt / nt) / (kb / nb):.2f}")
p_bn = stats.binom.sf(kt - 1, nt, kb / nb)
say(f"[homer] binomial p={p_bn:.2e}")

# puntuación de un sitio con la PWM verdadera (log-odds)
sitio = "ATGACTCA"
sc = sum(math.log2(TRUE[k, BASES.index(b)] / 0.25) for k, b in enumerate(sitio))
sitio2 = "ATGACTTA"
sc2 = sum(math.log2(TRUE[k, BASES.index(b)] / 0.25)
          for k, b in enumerate(sitio2))
mx = sum(math.log2(TRUE[k].max() / 0.25) for k in range(Wm))
say(f"[pwm] score {sitio}={sc:.2f} bits  {sitio2}={sc2:.2f}  max={mx:.2f}")
for k in range(Wm):
    say(f"[pwm] col {k + 1}: " + " ".join(
        f"{BASES[b]}={math.log2(TRUE[k, b] / 0.25):+.2f}" for b in range(4)))

(OUT / "cifras.txt").write_text("\n".join(LOG) + "\n")
