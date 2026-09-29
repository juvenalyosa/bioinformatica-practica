"""Genera datos de las figuras del capítulo 11 (RNA-seq).

Ejecutar desde libro/:  python3 figuras/cap11/generar.py
Escribe .dat en figuras/cap11/ e imprime (y guarda en cifras.txt) todas las
cifras que se citan en los ejemplos del texto (única fuente de verdad).
Todo es simulado con semilla fija; no se usan datos descargados.
"""
from pathlib import Path
import numpy as np
from scipy import stats, special, optimize

OUT = Path(__file__).parent
rng = np.random.default_rng(11)
LOG = []


def w(name, header, cols, fmt="{:.5g}"):
    rows = [" ".join(fmt.format(v) for v in r) for r in zip(*cols)]
    (OUT / name).write_text(header + "\n" + "\n".join(rows) + "\n")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


# =====================================================================
# 11.1  EM de isoformas (ejemplo de juguete)
# =====================================================================
L = np.array([1500.0, 1000.0, 2000.0])        # longitudes de T1, T2, T3
mu_frag = 200.0
Leff = L - mu_frag + 1                         # longitud efectiva simple
classes = [((0,), 120), ((1,), 30), ((2,), 200),
           ((0, 1), 250), ((0, 2), 150), ((0, 1, 2), 450)]
N = sum(c for _, c in classes)
alpha = np.ones(3) / 3
hist = [alpha.copy()]
for it in range(200):
    cnt = np.zeros(3)
    for C, n in classes:
        idx = list(C)
        wts = alpha[idx] / Leff[idx]
        cnt[idx] += n * wts / wts.sum()
    alpha = cnt / N
    hist.append(alpha.copy())
hist = np.array(hist)
say(f"[em] N={N} Leff={Leff}")
for k in range(4):
    say(f"[em] iter {k}: alpha=" + ", ".join(f"{a:.4f}" for a in hist[k]))
say("[em] final: alpha=" + ", ".join(f"{a:.4f}" for a in hist[-1]))
tau = hist[-1] / Leff
tau /= tau.sum()
say("[em] final tau (TPM/1e6)=" + ", ".join(f"{a:.4f}" for a in tau))
say("[em] lecturas asignadas final=" + ", ".join(f"{a*N:.1f}" for a in hist[-1]))
# iteración 1 detallada: reparto de la clase {T1,T2,T3}
a0 = hist[0]
wt = a0 / Leff
say("[em] iter1 reparto clase {1,2,3}: " + ", ".join(
    f"{450*x/wt.sum():.1f}" for x in wt))
wt12 = wt[[0, 1]]
say("[em] iter1 reparto clase {1,2}: " + ", ".join(
    f"{250*x/wt12.sum():.1f}" for x in wt12))
wt13 = wt[[0, 2]]
say("[em] iter1 reparto clase {1,3}: " + ", ".join(
    f"{150*x/wt13.sum():.1f}" for x in wt13))
# convergencia
d = np.abs(np.diff(hist, axis=0)).max(axis=1)
say(f"[em] iteraciones hasta cambio < 1e-4: {int(np.argmax(d < 1e-4)) + 1}")
it = np.arange(len(hist))[:41]
w("em.dat", "it a1 a2 a3", [it, *hist[:41].T])
# verosimilitud por iteración
def loglik(a):
    ll = 0.0
    for C, n in classes:
        idx = list(C)
        ll += n * np.log((a[idx] / Leff[idx]).sum())
    return ll
lls = np.array([loglik(a) for a in hist[:41]])
w("em_ll.dat", "it ll", [it, lls - lls[-1]])
say(f"[em] logL iter0={lls[0]:.2f} iter1={lls[1]:.2f} final={lls[-1]:.2f}")

# =====================================================================
# 11.1  RPKM vs TPM (dos muestras)
# =====================================================================
glen = np.array([1.0, 2.0, 4.0, 0.5])            # kb
# moléculas relativas de ARNm por gen en cada muestra
molA = np.array([10, 10, 10, 10.0])
molB = np.array([10, 10, 10, 40.0])               # B expresa mucho el gen corto
readsA = molA * glen; readsA = readsA / readsA.sum() * 10e6
readsB = molB * glen; readsB = readsB / readsB.sum() * 10e6
def rpkm(r, l):
    return r / (l * r.sum() / 1e6)
def tpm(r, l):
    x = r / l
    return x / x.sum() * 1e6
RA, RB = rpkm(readsA, glen), rpkm(readsB, glen)
TA, TB = tpm(readsA, glen), tpm(readsB, glen)
say("[tpm] lecturas A=", np.round(readsA).astype(int), " B=", np.round(readsB).astype(int))
say("[tpm] RPKM A=", np.round(RA, 1), " suma", round(RA.sum(), 1))
say("[tpm] RPKM B=", np.round(RB, 1), " suma", round(RB.sum(), 1))
say("[tpm] TPM A=", np.round(TA, 0), " suma", round(TA.sum(), 0))
say("[tpm] TPM B=", np.round(TB, 0), " suma", round(TB.sum(), 0))
w("rpkm.dat", "g A B", [np.arange(1, 5), RA, RB], "{:.2f}")
w("tpm.dat", "g A B", [np.arange(1, 5), TA / 1e3, TB / 1e3], "{:.2f}")

# =====================================================================
# 11.2  Poisson frente a binomial negativa: pmf y media-varianza
# =====================================================================
def nb_rvs(mu, a, size=None):
    """NB con media mu y dispersión a (Var = mu + a mu^2)."""
    r = 1.0 / a
    lam = rng.gamma(r, mu / r, size=size)
    return rng.poisson(lam)

mu0, a0_ = 20.0, 0.2
k = np.arange(0, 71)
pois = stats.poisson.pmf(k, mu0)
r_ = 1 / a0_
nbp = stats.nbinom.pmf(k, r_, r_ / (r_ + mu0))
w("pmf.dat", "k pois nb", [k, pois, nbp])
say(f"[pmf] mu=20 alpha=0.2: Var NB={mu0 + a0_ * mu0**2}, P(K=0) NB={nbp[0]:.4f},"
    f" P(K>=40) Pois={stats.poisson.sf(39, mu0):.2e}"
    f" NB={stats.nbinom.sf(39, r_, r_/(r_+mu0)):.3f}")

G = 1500
mus = np.exp(rng.uniform(np.log(0.5), np.log(2e4), G))
Yp = rng.poisson(np.repeat(mus[:, None], 6, 1))
Yn = nb_rvs(np.repeat(mus[:, None], 6, 1), 0.1)
def mv(Y):
    m, v = Y.mean(1), Y.var(1, ddof=1)
    ok = (m > 0) & (v > 0)
    return m[ok], v[ok]
mp, vp = mv(Yp); mn, vn = mv(Yn)
w("mv_pois.dat", "m v", [mp, vp], "{:.4g}")
w("mv_nb.dat", "m v", [mn, vn], "{:.4g}")
hi = mn > 1000
say(f"[mv] genes con media>1000: mediana Var/media Poisson="
    f"{np.median(vp[mp > 1000] / mp[mp > 1000]):.2f}, NB={np.median(vn[hi] / mn[hi]):.1f}")

# =====================================================================
# 11.2  Efecto de composición (dos muestras) y TMM / mediana de razones
# =====================================================================
Gc = 4000
lam = np.exp(rng.normal(3.5, 1.6, Gc))
lamB = lam.copy()
up = rng.choice(Gc, int(0.05 * Gc), replace=False)
lamB[up] *= 8.0
xa = rng.poisson(lam * 1.0)
xb = rng.poisson(lamB * 1.0)
# profundidades iguales en lecturas totales
xb = rng.poisson(lamB * xa.sum() / lamB.sum())
NA, NB_ = xa.sum(), xb.sum()
keep = (xa > 0) & (xb > 0)
M = np.log2((xb[keep] / NB_) / (xa[keep] / NA))
A = 0.5 * np.log2((xb[keep] / NB_) * (xa[keep] / NA))
# TMM (Robinson & Oshlack 2010), referencia = A
def tmm(obs, ref, logratio_trim=0.3, sum_trim=0.05):
    n_o, n_r = obs.sum(), ref.sum()
    ok = (obs > 0) & (ref > 0)
    o, r = obs[ok], ref[ok]
    Mg = np.log2((o / n_o) / (r / n_r))
    Ag = 0.5 * np.log2((o / n_o) * (r / n_r))
    v = (n_o - o) / n_o / o + (n_r - r) / n_r / r
    n = len(Mg)
    lo_m, hi_m = np.floor(n * logratio_trim) + 1, n + 1 - np.floor(n * logratio_trim)
    lo_a, hi_a = np.floor(n * sum_trim) + 1, n + 1 - np.floor(n * sum_trim)
    rm, ra = stats.rankdata(Mg), stats.rankdata(Ag)
    sel = (rm >= lo_m) & (rm <= hi_m) & (ra >= lo_a) & (ra <= hi_a)
    return 2 ** (np.sum(Mg[sel] / v[sel]) / np.sum(1 / v[sel]))
f_tmm = tmm(xb, xa)
# mediana de razones (DESeq)
Xab = np.vstack([xa, xb]).T.astype(float)
okg = (Xab > 0).all(1)
lg = np.log(Xab[okg])
geo = lg.mean(1)
sf = np.exp(np.median(lg - geo[:, None], axis=0))
sf_ratio = sf[1] / sf[0]
# en la escala de CPM: factor relativo sobre las proporciones
m_tmm = np.log2(f_tmm)
m_mor = np.log2(sf_ratio / (NB_ / NA))
frac_up = xb[up].sum() / NB_
say(f"[comp] N_A={NA} N_B={NB_} fracción de lecturas de B en genes 'up'={frac_up:.3f}"
    f" (en A: {xa[up].sum()/NA:.3f})")
say(f"[comp] mediana M de genes sin cambio (CPM)={np.median(np.log2((xb/NB_)/(xa/NA))[np.setdiff1d(np.where(keep)[0], up)]):.3f}")
say(f"[comp] TMM log2 f={m_tmm:.3f} (f={f_tmm:.3f});  mediana de razones log2={m_mor:.3f};"
    f" s_A={sf[0]:.3f} s_B={sf[1]:.3f}")
is_up = np.isin(np.where(keep)[0], up)
sub = rng.choice(len(M), 2500, replace=False)
A = A + np.log2(1e6)                 # en escala log2 CPM
w("comp_ns.dat", "A M", [A[sub][~is_up[sub]], M[sub][~is_up[sub]]], "{:.3f}")
w("comp_up.dat", "A M", [A[is_up], M[is_up]], "{:.3f}")
(OUT / "comp_lineas.tex").write_text(
    f"\\def\\caponcetmm{{{m_tmm:.3f}}}\n\\def\\caponcemor{{{m_mor:.3f}}}\n")
teo = np.log2(1 / (1 - 0 + (lam[up].sum() * 7) / lam.sum()))
say(f"[comp] log2 factor teórico de dilución = {teo:.3f}")

# ejemplo pequeño de mediana de razones (5 genes x 3 muestras)
ej = np.array([[500, 1100, 700], [80, 190, 105], [1200, 2600, 1550],
               [30, 64, 42], [10, 0, 9]], float)
okr = (ej > 0).all(1)
gm = np.exp(np.log(ej[okr]).mean(1))
rat = ej[okr] / gm[:, None]
sfe = np.median(rat, axis=0)
say("[mor] medias geométricas=", np.round(gm, 2))
say("[mor] razones=\n", np.round(rat, 3))
say("[mor] factores de tamaño=", np.round(sfe, 3))

# =====================================================================
# 11.2-11.3  Experimento simulado 3 vs 3 y análisis tipo DESeq2
# =====================================================================
G = 10000
glen_g = np.exp(rng.normal(np.log(2000), 0.8, G))        # longitud (pb)
rate = np.exp(rng.normal(-4.2, 1.2, G))                  # expresión por pb
q0 = rate * glen_g                                      # media base (ctrl)
a_tr = lambda m: 0.05 + 1.0 / m                          # tendencia verdadera
disp = a_tr(q0) * np.exp(rng.normal(0, 0.3, G))
de = rng.random(G) < 0.10
lfc = np.zeros(G)
lfc[de] = (0.5 + rng.exponential(0.8, de.sum())) * rng.choice([-1, 1], de.sum())
s_true = np.array([0.8, 1.2, 1.0, 0.9, 1.3, 1.1])
cond = np.array([0, 0, 0, 1, 1, 1])
mu = q0[:, None] * 2 ** (lfc[:, None] * cond[None, :]) * s_true[None, :]
Y = nb_rvs(mu, disp[:, None])
keepg = Y.sum(1) > 0
Y, de, lfc, glen_g, disp = Y[keepg], de[keepg], lfc[keepg], glen_g[keepg], disp[keepg]
G = Y.shape[0]
say(f"[sim] genes con alguna lectura={G}; DE verdaderos={de.sum()};"
    f" lecturas por muestra=", Y.sum(0))
# factores de tamaño
okg = (Y > 0).all(1)
lgy = np.log(Y[okg])
sf = np.exp(np.median(lgy - lgy.mean(1)[:, None], axis=0))
say("[sim] factores de tamaño estimados=", np.round(sf, 3),
    " verdaderos (reescalados)=", np.round(s_true / np.exp(np.log(s_true).mean()), 3))
Yn = Y / sf[None, :]
base = Yn.mean(1)
X = np.column_stack([np.ones(6), cond])

def nb_ll(y, m, a):
    r = 1.0 / a
    return (special.gammaln(y + r) - special.gammaln(r) - special.gammaln(y + 1)
            + r * np.log(r / (r + m)) + y * np.log(m / (r + m)))

# medias por grupo (aprox. MLE) para estimar dispersión gen a gen
qg = np.stack([Yn[:, cond == c].mean(1) for c in (0, 1)], 1)
mu_hat = np.maximum(qg[:, cond] * sf[None, :], 1e-8)
grid = np.exp(np.linspace(np.log(1e-4), np.log(20), 300))
LLg = np.stack([nb_ll(Y, mu_hat, a).sum(1) for a in grid], 1)       # G x grid
# ajuste Cox-Reid: -0.5 log det(X^T W X)
def cr_adj(a):
    Wm = mu_hat / (1 + a * mu_hat)
    s00 = Wm.sum(1); s01 = (Wm * cond).sum(1)
    return -0.5 * np.log(s00 * s01 - s01 ** 2)
LLg += np.stack([cr_adj(a) for a in grid], 1)
a_gw = grid[LLg.argmax(1)]
# tendencia a0 + a1/mu: GLM gamma con enlace identidad (devianza gamma)
fitm = (base > 1) & (a_gw > 1e-3) & (a_gw < grid[-1])
def gdev(p):
    f = np.abs(p[0]) + np.abs(p[1]) / base[fitm]
    r = a_gw[fitm] / f
    return 2 * np.sum(r - np.log(r) - 1)
pt = optimize.minimize(gdev, [0.1, 1.0], method="Nelder-Mead").x
a0h, a1h = abs(pt[0]), abs(pt[1])
a_fit = a0h + a1h / base
say(f"[disp] tendencia ajustada: a0={a0h:.4f} a1={a1h:.3f} (verdad: 0,05 y 1)")
# varianza previa (log): var residual - varianza de muestreo esperada
res = np.log(a_gw[fitm]) - np.log(a_fit[fitm])
mad = stats.median_abs_deviation(res, scale="normal")
samp = special.polygamma(1, (6 - 2) / 2)
s2p = max(mad ** 2 - samp, 0.25)
say(f"[disp] sd robusta residuos log={mad:.3f}; var muestreo={samp:.3f}; var previa={s2p:.3f}")
logprior = -0.5 * (np.log(grid)[None, :] - np.log(a_fit)[:, None]) ** 2 / s2p
a_map = grid[(LLg + logprior).argmax(1)]
# valores atípicos de dispersión: se conserva la estimación gen a gen
outl = np.log(a_gw) > np.log(a_fit) + 2 * np.sqrt(s2p)
a_final = np.where(outl, a_gw, a_map)
say(f"[disp] genes con dispersión atípica (>2 sd sobre la tendencia)={outl.sum()}")
# figura de dispersión (submuestra)
subd = rng.choice(np.where(base > 0.5)[0], 2500, replace=False)
w("disp_gw.dat", "m a", [base[subd], a_gw[subd]], "{:.4g}")
w("disp_map.dat", "m a", [base[subd], a_final[subd]], "{:.4g}")
xs = np.exp(np.linspace(np.log(0.5), np.log(base.max()), 60))
w("disp_fit.dat", "m a", [xs, a0h + a1h / xs], "{:.4g}")
w("disp_true.dat", "m a", [xs, 0.05 + 1 / xs], "{:.4g}")
for gi in [int(np.argmin(np.abs(base - 20)))]:
    say(f"[disp] gen ej. media={base[gi]:.1f}: gen a gen={a_gw[gi]:.3f},"
        f" tendencia={a_fit[gi]:.3f}, MAP={a_map[gi]:.3f}, verdad={disp[gi]:.3f}")
say(f"[disp] error cuadrático medio log: gen a gen="
    f"{np.mean((np.log(a_gw)-np.log(disp))[base>1]**2):.3f}  MAP="
    f"{np.mean((np.log(a_map)-np.log(disp))[base>1]**2):.3f}")

# GLM NB por IRLS (dos coeficientes, offset log s_j) y prueba de Wald
def irls(Y, a, iters=30):
    off = np.log(sf)[None, :]
    b = np.stack([np.log(np.maximum(qg[:, 0], 0.1)),
                  np.log(np.maximum(qg[:, 1], 0.1)) - np.log(np.maximum(qg[:, 0], 0.1))], 1)
    for _ in range(iters):
        eta = b @ X.T + off
        m = np.exp(eta)
        Wt = m / (1 + a[:, None] * m)
        z = eta - off + (Y - m) / m
        s00 = Wt.sum(1); s01 = (Wt * cond).sum(1); s11 = s01
        r0 = (Wt * z).sum(1); r1 = (Wt * z * cond).sum(1)
        det = s00 * s11 - s01 ** 2
        b = np.stack([(s11 * r0 - s01 * r1) / det, (s00 * r1 - s01 * r0) / det], 1)
        b = np.clip(b, -30, 30)
    m = np.exp(b @ X.T + off)
    Wt = m / (1 + a[:, None] * m)
    s00 = Wt.sum(1); s01 = (Wt * cond).sum(1)
    det = s00 * s01 - s01 ** 2
    se1 = np.sqrt(s00 / det)
    return b, se1, m
b, se, mfit = irls(Y, a_final)
lfc_hat = b[:, 1] / np.log(2)
se2 = se / np.log(2)
wald = lfc_hat / se2
pval = 2 * stats.norm.sf(np.abs(wald))
# LRT (modelo completo vs reducido) para comparar
m_red = np.maximum(Yn.mean(1), 1e-8)[:, None] * sf[None, :]
lrt = 2 * (nb_ll(Y, mfit, a_final[:, None]).sum(1) - nb_ll(Y, m_red, a_final[:, None]).sum(1))
p_lrt = stats.chi2.sf(np.maximum(lrt, 0), 1)
say(f"[wald] correlación log10 p Wald vs LRT="
    f"{np.corrcoef(np.log10(pval+1e-300), np.log10(p_lrt+1e-300))[0,1]:.4f}")

def bh(p):
    m = len(p); o = np.argsort(p)
    q = p[o] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(q, 1)
    return out
padj = bh(pval)
sig = padj < 0.1
fdp = (sig & ~de).sum() / max(sig.sum(), 1)
say(f"[de] p<0,05 sin corregir: {np.sum(pval<0.05)} (falsos: {np.sum((pval<0.05)&~de)})")
say(f"[de] BH q<0,1: {sig.sum()} genes; falsos={np.sum(sig & ~de)} FDP={fdp:.3f};"
    f" potencia={np.sum(sig & de)/de.sum():.3f}")
bonf = pval < 0.1 / G
say(f"[de] Bonferroni FWER 0,1: {bonf.sum()} genes; falsos={np.sum(bonf & ~de)}")
# Storey pi0
lam_s = 0.5
pi0 = np.mean(pval > lam_s) / (1 - lam_s)
say(f"[pi0] pi0 estimado (lambda=0,5)={pi0:.3f}; verdad={np.mean(~de):.3f}; m={G}")
qv = np.minimum(1, pi0 * padj)
ff = base >= 5
pi0f = np.mean(pval[ff] > lam_s) / (1 - lam_s)
say(f"[pi0] tras filtrar media>=5: pi0={pi0f:.3f}; verdad={np.mean(~de[ff]):.3f}; m={ff.sum()}")
say(f"[pi0] fracción de p>0,95 entre genes con media<2: {np.mean(pval[base<2]>0.95):.3f}")
say(f"[pi0] q-value<0,1: {np.sum(qv<0.1)} genes")
# histograma de p (20 barras) separando nulos / no nulos
edges = np.linspace(0, 1, 21)
h0, _ = np.histogram(pval[~de], edges)
h1, _ = np.histogram(pval[de], edges)
w("phist.dat", "x h0 h1", [edges[:-1] + 0.025, h0, h1], "{:.4g}")
(OUT / "phist.tex").write_text(
    f"\\def\\caponcepinivel{{{pi0 * G / 20:.2f}}}\n\\def\\caponcepizero{{{pi0:.2f}}}\n")
say(f"[pi0] altura nula esperada por barra={pi0*G/20:.1f}")
# BH escalera (primeros 1200 p ordenados)
o = np.argsort(pval)
ps = pval[o][:1200]
ii = np.arange(1, 1201)
kbh = np.max(np.where(pval[o] <= 0.1 * np.arange(1, G + 1) / G)[0]) + 1
dd = de[o][:1200]
w("bh_de.dat", "i p", [ii[dd], ps[dd]], "{:.4g}")
w("bh_null.dat", "i p", [ii[~dd], ps[~dd]], "{:.4g}")
say(f"[bh] k* (último p_(i) <= i*0,1/m) = {kbh}; p_(k*)={pval[o][kbh-1]:.3e};"
    f" umbral={0.1*kbh/G:.3e}")
# filtrado independiente
qs = np.linspace(0, 0.8, 33)
nrej, nfd = [], []
for qq in qs:
    thr = np.quantile(base, qq)
    f = base >= thr
    pa = bh(pval[f])
    nrej.append(np.sum(pa < 0.1))
    nfd.append(np.sum((pa < 0.1) & ~de[f]))
w("filtro.dat", "q n", [qs, nrej], "{:.4g}")
ib = int(np.argmax(nrej))
say(f"[filtro] sin filtro={nrej[0]}; máximo={nrej[ib]} al cuantil {qs[ib]:.3f}"
    f" (umbral media={np.quantile(base, qs[ib]):.2f}); FDP allí={nfd[ib]/max(nrej[ib],1):.3f}")
# contracción del LFC (prior normal, al estilo de DESeq2 2014)
use = base > 5
sp = np.quantile(np.abs(lfc_hat[use]), 0.95) / stats.norm.ppf(0.975)
lfc_shr = lfc_hat * sp ** 2 / (sp ** 2 + se2 ** 2)
say(f"[shr] sigma prior={sp:.3f}; ECM LFC MLE={np.mean((lfc_hat-lfc)[base>1]**2):.3f}"
    f" contraído={np.mean((lfc_shr-lfc)[base>1]**2):.3f} (genes con media>1)")
gi = int(np.argmax(np.where((base > 1) & (base < 5) & (se2 < 5), np.abs(lfc_hat) * (~de), 0)))
say(f"[shr] gen ruidoso: media={base[gi]:.2f} LFC MLE={lfc_hat[gi]:.2f}"
    f" SE={se2[gi]:.2f} contraído={lfc_shr[gi]:.2f} p={pval[gi]:.3f}")
subm = np.concatenate([rng.choice(np.where(~sig & (base > 0.3))[0], 2200, replace=False),
                       np.where(sig)[0]])
subm = np.unique(subm)
for tag, v in (("mle", lfc_hat), ("shr", lfc_shr)):
    s_ = subm
    w(f"ma_{tag}_ns.dat", "a m", [base[s_][~sig[s_]], np.clip(v[s_][~sig[s_]], -6, 6)], "{:.4g}")
    w(f"ma_{tag}_sig.dat", "a m", [base[s_][sig[s_]], np.clip(v[s_][sig[s_]], -6, 6)], "{:.4g}")
lp = -np.log10(np.maximum(pval, 1e-60))
cat_up = sig & (lfc_hat > 1)
cat_dn = sig & (lfc_hat < -1)
cat_ns = ~(cat_up | cat_dn)
subv = np.unique(np.concatenate([rng.choice(np.where(cat_ns)[0], 2500, replace=False),
                                 np.where(~cat_ns)[0]]))
for tag, c in (("ns", cat_ns), ("up", cat_up), ("dn", cat_dn)):
    s_ = subv[c[subv]]
    w(f"vol_{tag}.dat", "l p", [np.clip(lfc_hat[s_], -8, 8), np.minimum(lp[s_], 60)], "{:.4g}")
say(f"[vol] up (q<0,1, LFC>1)={cat_up.sum()} down={cat_dn.sum()}; -log10 p umbral BH"
    f"={-np.log10(0.1*kbh/G):.2f}")

# =====================================================================
# 11.4  Sesgo de longitud (PWF), hipergeométrica y GSEA
# =====================================================================
# PWF: proporción de genes DE detectados por bin de longitud
bins = np.quantile(glen_g, np.linspace(0, 1, 21))
bi = np.clip(np.digitize(glen_g, bins) - 1, 0, 19)
cen = np.array([np.median(glen_g[bi == j]) for j in range(20)])
prop = np.array([sig[bi == j].mean() for j in range(20)])
w("pwf.dat", "l p", [cen, prop], "{:.4g}")
# ajuste logístico en log-longitud
def nll(p):
    z = p[0] + p[1] * np.log(glen_g)
    return -np.sum(sig * z - np.log1p(np.exp(z)))
pp = optimize.minimize(nll, [-5, 0.5]).x
xl = np.exp(np.linspace(np.log(cen[0]), np.log(cen[-1]), 50))
w("pwf_fit.dat", "l p", [xl, special.expit(pp[0] + pp[1] * np.log(xl))], "{:.4g}")
say(f"[pwf] proporción DE bin corto={prop[0]:.3f} bin largo={prop[-1]:.3f};"
    f" pendiente logit por log-longitud={pp[1]:.2f}")
# término GO sesgado a genes largos, sin enriquecimiento real
long_rank = stats.rankdata(glen_g) / G
cand = np.where(long_rank > 0.8)[0]
nde_t = int(round(150 * de.mean()))
term = np.concatenate([rng.choice(cand[de[cand]], nde_t, replace=False),
                       rng.choice(cand[~de[cand]], 150 - nde_t, replace=False)])
Nn, Kk, nn = G, len(term), int(sig.sum())
kk = int(sig[term].sum())
pv_h = stats.hypergeom.sf(kk - 1, Nn, Kk, nn)
say(f"[hiper] N={Nn} K={Kk} n={nn} k={kk} esperado={Kk*nn/Nn:.2f}"
    f" p(hipergeom)={pv_h:.2e}  DE verdaderos en término={de[term].sum()} (esperado {Kk*de.mean():.1f})")
# tabla 2x2 y Fisher
tab = np.array([[kk, Kk - kk], [nn - kk, Nn - Kk - nn + kk]])
say(f"[hiper] tabla={tab.tolist()} Fisher p unilateral={stats.fisher_exact(tab, 'greater')[1]:.2e}"
    f" OR={stats.fisher_exact(tab)[0]:.2f}")
kx = np.arange(0, 41)
w("hiper.dat", "k p", [kx, stats.hypergeom.pmf(kx, Nn, Kk, nn)], "{:.4g}")
(OUT / "hiper.tex").write_text(f"\\def\\caponcekobs{{{kk}}}\n")
# nulo con muestreo ponderado por PWF (estilo goseq, remuestreo)
pwf = special.expit(pp[0] + pp[1] * np.log(glen_g))
wts = pwf / pwf.sum()
sims = np.array([np.isin(rng.choice(G, nn, replace=False, p=wts), term).sum()
                 for _ in range(4000)])
p_w = (np.sum(sims >= kk) + 1) / (len(sims) + 1)
say(f"[goseq] media nula ponderada={sims.mean():.2f}; p remuestreo={p_w:.3f}")
hw, _ = np.histogram(sims, np.arange(-0.5, 41.5))
w("hiper_w.dat", "k p", [kx, hw / hw.sum()], "{:.4g}")

# GSEA: conjunto de 80 genes enriquecido en genes sobreexpresados
stat_rank = wald.copy()
order = np.argsort(-stat_rank)
upg = np.where(de & (lfc > 0))[0]
gset = np.concatenate([rng.choice(upg, 28, replace=False),
                       rng.choice(np.where(~de)[0], 52, replace=False)])
def running(order, inset, r, p=1.0):
    hit = inset[order]
    wv = np.abs(r[order]) ** p
    ph = np.cumsum(np.where(hit, wv, 0)) / np.sum(wv[hit])
    pm = np.cumsum(~hit) / np.sum(~hit)
    return ph - pm
inset = np.zeros(G, bool); inset[gset] = True
rs = running(order, inset, stat_rank)
ES = rs[np.argmax(np.abs(rs))]
imax = int(np.argmax(np.abs(rs)))
nulls = []
for _ in range(1000):
    ins = np.zeros(G, bool); ins[rng.choice(G, len(gset), replace=False)] = True
    rr = running(order, ins, stat_rank)
    nulls.append(rr[np.argmax(np.abs(rr))])
nulls = np.array(nulls)
pos = nulls[nulls > 0]
p_es = (np.sum(pos >= ES) + 1) / (len(pos) + 1)
NES = ES / pos.mean()
lead = np.sum(inset[order][:imax + 1])
say(f"[gsea] tamaño={len(gset)}, ES={ES:.3f} en posición {imax+1} de {G};"
    f" NES={NES:.2f}; p (perm. de genes)={p_es:.4f}; genes líderes={lead}")
rs_unw = running(order, inset, stat_rank, p=0)
say(f"[gsea] ES no ponderado (p=0, KS)={rs_unw[np.argmax(np.abs(rs_unw))]:.3f}")
step = 5
xi = np.arange(0, G, step)
w("gsea_rs.dat", "i rs", [xi + 1, rs[xi]], "{:.4g}")
w("gsea_hits.dat", "i", [np.where(inset[order])[0] + 1], "{:.0f}")
w("gsea_metric.dat", "i r", [xi + 1, np.clip(stat_rank[order][xi], -15, 15)], "{:.4g}")
(OUT / "gsea.tex").write_text(
    f"\\def\\caponceesx{{{imax+1}}}\n\\def\\caponceesy{{{ES:.3f}}}\n\\def\\caponceG{{{G}}}\n")

(OUT / "cifras.txt").write_text("\n".join(LOG) + "\n")
