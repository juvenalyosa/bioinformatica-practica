"""Genera datos y figuras del capítulo 17 (machine learning en bioinformática).

Ejecutar desde libro/:  python3 figuras/cap17/generar.py

Requisitos: numpy, scikit-learn, torch, umap-learn, transformers, biopython,
matplotlib. La lección 17.3 usa el modelo ESM-2 de 650 M de parámetros
(facebook/esm2_t33_650M_UR50D) desde la caché local de HuggingFace
(~/.cache/huggingface); si no está, transformers lo descarga (~2,5 GB).
Descarga además 1UBQ (RCSB) y secuencias de UniProt a una caché temporal.
Escribe .dat/.tex/.pdf en figuras/cap17/ e imprime las cifras citadas.
"""
from pathlib import Path
import json, math, tempfile, time, warnings, subprocess, urllib.parse
import numpy as np
import requests

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent
CACHE = Path(tempfile.gettempdir()) / "cap17_cache"
CACHE.mkdir(exist_ok=True)
rng = np.random.default_rng(17)
S = requests.Session()
S.headers["User-Agent"] = "bioinfo-libro-cap17 (mailto:juvenal.yosa@gmail.com)"
NUM = {}

COL = dict(azul="#2A78D6", naranja="#EB6834", aqua="#1BAF7A", amarillo="#EDA100",
           magenta="#E87BA4", verde="#008300", violeta="#4A3AA7", rojo="#E34948",
           tinta="#0B0B0B", tinta2="#52514E", gris="#898781", rejilla="#E1E0D9",
           azulnoche="#0D366B", azulpalido="#CDE2FB", papel="#F0EFEC")
NUCCOL = {"A": "nucA", "C": "nucC", "G": "nucG", "T": "nucT"}


def w(name, text):
    (OUT / name).write_text(text)


def dat(name, header, rows):
    def f(v):
        return f"{v:.5g}" if isinstance(v, (float, np.floating)) else str(v)
    w(name, header + "\n" + "\n".join(" ".join(f(v) for v in r) for r in rows) + "\n")


def say(k, v):
    NUM[k] = v
    print(f"{k:40s} {v}")


def fetch(url, name):
    p = CACHE / name
    if not p.exists() or p.stat().st_size == 0:
        for k in range(4):
            try:
                r = S.get(url, timeout=90)
                r.raise_for_status()
                p.write_bytes(r.content)
                break
            except Exception:
                time.sleep(2 + 2 * k)
    return p


# =====================================================================
# 17.1  ML clásico
# =====================================================================
from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (KFold, GroupKFold, StratifiedKFold,
                                     cross_val_score, cross_validate)
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                             average_precision_score)


def kmers(k, alf="ACGT"):
    return ["".join(p) for p in product(alf, repeat=k)]


def spectrum(s, k, alf="ACGT"):
    idx = {u: i for i, u in enumerate(kmers(k, alf))}
    v = np.zeros(len(idx))
    for i in range(len(s) - k + 1):
        u = s[i:i + k]
        if u in idx:
            v[idx[u]] += 1
    return v


# ---- A1. one-hot y espectro de 2-mers --------------------------------
x_seq, y_seq = "GATTACAGAT", "TACAGATTAC"
fx, fy = spectrum(x_seq, 2), spectrum(y_seq, 2)
say("kmer: x", x_seq)
say("kmer: phi(x) no nulos", {u: int(c) for u, c in zip(kmers(2), fx) if c})
say("kmer: phi(y) no nulos", {u: int(c) for u, c in zip(kmers(2), fy) if c})
say("kmer: K(x,y)", float(fx @ fy))
say("kmer: K(x,x), K(y,y)", (float(fx @ fx), float(fy @ fy)))
say("kmer: K normalizado", round(float(fx @ fy / math.sqrt((fx @ fx) * (fy @ fy))), 4))

t = [r"\begin{tikzpicture}[font=\sffamily\scriptsize]"]
cs = 0.5
for i, b in enumerate(x_seq):
    t.append(rf"\node[nt={b},minimum size=4.6mm] at ({i*cs:.2f},0.75) {{{b}}};")
    t.append(rf"\node[text=gris] at ({i*cs:.2f},1.25) {{{i+1}}};")
for r, base in enumerate("ACGT"):
    yy = -r * cs
    t.append(rf"\node[anchor=east,font=\ttfamily\bfseries\small,text={NUCCOL[base]}] at (-0.35,{yy:.2f}) {{{base}}};")
    for i, b in enumerate(x_seq):
        on = b == base
        fill = f"{NUCCOL[base]}!75" if on else "white"
        txt = r"\color{white}\bfseries 1" if on else r"\color{base}0"
        t.append(rf"\node[draw=rejilla,fill={fill},minimum size={cs}cm,inner sep=0pt] at ({i*cs:.2f},{yy:.2f}) {{{txt}}};")
t.append(rf"\draw[decorate,decoration={{brace,amplitude=4pt,mirror}},tinta2] (-0.25,{-3*cs-0.35:.2f}) -- ({(len(x_seq)-1)*cs+0.25:.2f},{-3*cs-0.35:.2f}) node[midway,below=5pt,etiqueta] {{matriz one-hot $X\in\{{0,1\}}^{{4\times L}}$}};")
t.append(r"\node[etiqueta,font=\sffamily\bfseries\small,text=tinta] at (2.25,1.85) {(a) codificación one-hot};")
coords = ",".join(kmers(2))
pts = " ".join(f"({u},{int(c)})" for u, c in zip(kmers(2), fx))
t.append(r"\begin{axis}[curso, at={(6.2cm,-1.9cm)}, width=7.0cm, height=4.3cm, ybar, bar width=5pt,"
         r" symbolic x coords={" + coords + r"}, xtick=data, ymin=0, ymax=3.4, ytick={0,1,2,3},"
         r" x tick label style={rotate=90,font=\ttfamily\tiny}, ylabel={conteo},"
         r" title={(b) espectro de 2-mers $\Phi_2(x)$}, enlarge x limits=0.04, grid=none]")
t.append(r"\addplot[fill=azul!70,draw=azul] coordinates {" + pts + "};")
t.append(r"\end{axis}")
t.append(r"\end{tikzpicture}")
w("onehot_kmer.tex", "\n".join(t) + "\n")

# ---- A2. un paso de gradiente de regresión logística -----------------
Xs = np.array([[2, 1], [1, 0], [0, 1], [0, 0]], float)
ys = np.array([1, 1, 0, 0], float)
Xb = np.c_[np.ones(4), Xs]
wv = np.zeros(3)
sig = lambda z: 1 / (1 + np.exp(-z))
p0 = sig(Xb @ wv)
L0 = -np.mean(ys * np.log(p0) + (1 - ys) * np.log(1 - p0))
g = Xb.T @ (p0 - ys) / 4
w1 = wv - 1.0 * g
p1 = sig(Xb @ w1)
L1 = -np.mean(ys * np.log(p1) + (1 - ys) * np.log(1 - p1))
say("logreg: gradiente", g.round(4).tolist())
say("logreg: w1", w1.round(4).tolist())
say("logreg: p1", p1.round(4).tolist())
say("logreg: L0, L1", (round(L0, 4), round(L1, 4)))

# ---- A3. frontera de decisión: regresión logística vs SVM ------------
n = 70
gc0 = rng.normal(0.44, 0.055, n); oe0 = rng.normal(0.36, 0.12, n)
gc1 = rng.normal(0.56, 0.055, n); oe1 = rng.normal(0.66, 0.14, n)
F = np.c_[np.r_[gc0, gc1], np.r_[oe0, oe1]]
yF = np.r_[np.zeros(n), np.ones(n)].astype(int)
mu, sd = F.mean(0), F.std(0)
Z = (F - mu) / sd
lr = LogisticRegression(C=1.0).fit(Z, yF)
svm = SVC(kernel="linear", C=0.5).fit(Z, yF)


def raw_line(wz, bz, level, x1):
    a = wz / sd
    c = bz - np.sum(wz * mu / sd)
    return (level - c - a[0] * x1) / a[1]


xs_ = np.array([0.28, 0.72])
rows = []
for lab, (wz, bz) in {"lr": (lr.coef_[0], lr.intercept_[0]),
                      "svm": (svm.coef_[0], svm.intercept_[0])}.items():
    levels = {"lr": [0, math.log(9), -math.log(9)], "svm": [0, 1, -1]}[lab]
    for j, lev in enumerate(levels):
        yv = raw_line(wz, bz, lev, xs_)
        dat(f"frontera_{lab}{j}.dat", "x y", zip(xs_, yv))
dat("frontera_pts.dat", "gc oe clase", [(a, b, c) for (a, b), c in zip(F, yF)])
sv = svm.support_
dat("frontera_sv.dat", "gc oe", [tuple(F[i]) for i in sv])
say("frontera: acc LR (entrenamiento)", round(lr.score(Z, yF), 3))
say("frontera: acc SVM (entrenamiento)", round(svm.score(Z, yF), 3))
say("frontera: n vectores soporte", len(sv))
say("frontera: CV10 LR / SVM", (round(cross_val_score(LogisticRegression(), Z, yF, cv=10).mean(), 3),
                                round(cross_val_score(SVC(kernel='linear', C=0.5), Z, yF, cv=10).mean(), 3)))

# ---- A4. sesgo-varianza: profundidad de árbol ------------------------
from sklearn.datasets import make_moons
Xm, ym = make_moons(n_samples=400, noise=0.33, random_state=17)
rows = []
cv10 = StratifiedKFold(10, shuffle=True, random_state=17)
for d in range(1, 16):
    res = cross_validate(DecisionTreeClassifier(max_depth=d, random_state=0), Xm, ym,
                         cv=cv10, return_train_score=True)
    rows.append((d, 1 - res["train_score"].mean(), 1 - res["test_score"].mean(),
                 res["test_score"].std()))
dat("sesgo_varianza.dat", "d train cv cvsd", rows)
best = min(rows, key=lambda r: r[2])
rf_err = 1 - cross_val_score(RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                    random_state=0), Xm, ym, cv=cv10).mean()
say("sv: mejor profundidad, error CV", (best[0], round(best[2], 3)))
say("sv: error CV profundidad 15 / train", (round(rows[-1][2], 3), round(rows[-1][1], 3)))
say("sv: error CV random forest", round(rf_err, 3))
w("sesgo_varianza_rf.tex", rf"\def\capdiecisieteRF{{{rf_err:.4f}}}" + "\n")

# ---- A5. fuga de información por homología ---------------------------
nfam, nmem, Lf, mut = 60, 15, 200, 0.15
seqs, groups, labels = [], [], []
fam_lab = rng.permutation(np.r_[np.zeros(nfam // 2), np.ones(nfam // 2)]).astype(int)
for f in range(nfam):
    anc = rng.choice(list("ACGT"), Lf)
    for m in range(nmem):
        s = anc.copy()
        pos = rng.random(Lf) < mut
        s[pos] = rng.choice(list("ACGT"), pos.sum())
        seqs.append("".join(s)); groups.append(f); labels.append(fam_lab[f])
Xk = np.array([spectrum(s, 3) for s in seqs]); Xk /= Xk.sum(1, keepdims=True)
yk, gk = np.array(labels), np.array(groups)
modelos = {"regresión logística": LogisticRegression(C=10, max_iter=5000),
           "random forest": RandomForestClassifier(n_estimators=300, random_state=0),
           "1-NN": KNeighborsClassifier(1)}
rows = []
for j, (nm, mdl) in enumerate(modelos.items()):
    a_r = cross_val_score(mdl, Xk, yk, cv=KFold(5, shuffle=True, random_state=17)).mean()
    a_g = cross_val_score(mdl, Xk, yk, cv=GroupKFold(5), groups=gk).mean()
    rows.append((j, a_r, a_g))
    say(f"fuga: {nm} aleatoria / por familia", (round(a_r, 3), round(a_g, 3)))
dat("fuga.dat", "i aleatoria familia", rows)

# ---- A6. ROC vs PR con desbalance ------------------------------------
def curvas(npos, nneg, tag):
    sp = rng.normal(1.5, 1, npos); sn = rng.normal(0, 1, nneg)
    yy = np.r_[np.ones(npos), np.zeros(nneg)]; ss = np.r_[sp, sn]
    fpr, tpr, _ = roc_curve(yy, ss)
    pr, rc, _ = precision_recall_curve(yy, ss)
    k = np.unique(np.linspace(0, len(fpr) - 1, 160).astype(int))
    dat(f"roc_{tag}.dat", "fpr tpr", zip(fpr[k], tpr[k]))
    k = np.unique(np.linspace(0, len(pr) - 1, 220).astype(int))
    dat(f"pr_{tag}.dat", "rec prec", zip(rc[k], pr[k]))
    return roc_auc_score(yy, ss), average_precision_score(yy, ss)


auc_b, ap_b = curvas(2000, 2000, "bal")
auc_d, ap_d = curvas(200, 19800, "des")
say("roc: AUC bal / des", (round(auc_b, 3), round(auc_d, 3)))
say("pr: AP bal / des", (round(ap_b, 3), round(ap_d, 3)))
say("roc: AUC teórico", round(0.5 * (1 + math.erf(1.5 / 2)), 3))
prev, tpr_, fpr_ = 0.01, 0.90, 0.05
say("ejemplo: precisión prev 1%", round(tpr_ * prev / (tpr_ * prev + fpr_ * (1 - prev)), 4))

# =====================================================================
# 17.2  CNN para ADN (PyTorch)
# =====================================================================
import torch
import torch.nn as nn
torch.manual_seed(17)
BASES = "ACGT"
PFM = np.array([  # motivo tipo AP-1 (TGA[CG]TCA)
    [0.03, 0.03, 0.03, 0.91], [0.03, 0.03, 0.91, 0.03], [0.91, 0.03, 0.03, 0.03],
    [0.03, 0.47, 0.47, 0.03], [0.03, 0.03, 0.03, 0.91], [0.03, 0.91, 0.03, 0.03],
    [0.91, 0.03, 0.03, 0.03]])
Lc, Nc = 100, 6000


def onehot(s):
    return np.array([[c == b for c in s] for b in BASES], np.float32)


def sim_seq(pos):
    s = rng.choice(list(BASES), Lc)
    at = -1
    if pos:
        at = rng.integers(0, Lc - len(PFM))
        for j, p in enumerate(PFM):
            s[at + j] = rng.choice(list(BASES), p=p)
    return "".join(s), at


yc = rng.integers(0, 2, Nc)
sims = [sim_seq(v) for v in yc]
Xc = torch.tensor(np.stack([onehot(s) for s, _ in sims]))
Yc = torch.tensor(yc, dtype=torch.float32)
itr, iva, ite = np.arange(0, 4000), np.arange(4000, 5000), np.arange(5000, 6000)


class CNN(nn.Module):
    def __init__(self, nf=8, k=11):
        super().__init__()
        self.conv = nn.Conv1d(4, nf, k)
        self.fc = nn.Linear(nf, 1)

    def forward(self, x):
        h = torch.relu(self.conv(x))
        return self.fc(h.max(dim=2).values).squeeze(1)


net = CNN()
say("cnn: parámetros", sum(p.numel() for p in net.parameters()))
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
lossf = nn.BCEWithLogitsLoss()
hist = []
for ep in range(1, 41):
    net.train()
    perm = torch.tensor(rng.permutation(itr))
    for b in range(0, len(perm), 64):
        ib = perm[b:b + 64]
        opt.zero_grad()
        loss = lossf(net(Xc[ib]), Yc[ib])
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        ltr = lossf(net(Xc[itr]), Yc[itr]).item()
        lva = lossf(net(Xc[iva]), Yc[iva]).item()
    hist.append((ep, ltr, lva))
dat("cnn_hist.dat", "ep train val", hist)
with torch.no_grad():
    lte = net(Xc[ite]).numpy()
say("cnn: pérdida final train/val", (round(hist[-1][1], 3), round(hist[-1][2], 3)))
say("cnn: AUROC test", round(roc_auc_score(yc[ite], lte), 3))
say("cnn: exactitud test", round(((lte > 0) == yc[ite]).mean(), 3))

# filtro más relevante -> logo por ventanas de máxima activación
Wfc = net.fc.weight.detach().numpy()[0]
fbest = int(np.argmax(Wfc))
with torch.no_grad():
    act = torch.relu(net.conv(Xc[np.r_[iva, ite]])).numpy()[:, fbest, :]
thr = 0.5 * act.max()
seqs_c = [sims[i][0] for i in np.r_[iva, ite]]
cnt = np.ones((11, 4)) * 0.1
nwin = 0
for s, a in zip(seqs_c, act):
    j = int(a.argmax())
    if a[j] > thr:
        nwin += 1
        for q, c in enumerate(s[j:j + 11]):
            cnt[q, BASES.index(c)] += 1
pfm_learn = cnt / cnt.sum(1, keepdims=True)
say("cnn: filtro elegido, ventanas", (fbest, nwin))


def ic_col(p):
    return 2 + np.sum(p * np.log2(np.clip(p, 1e-9, 1)))


# alinear el logo aprendido con el motivo verdadero (desfase de máx. IC)
ics = np.array([ic_col(p) for p in pfm_learn])
off = int(np.argmax([ics[o:o + 7].sum() for o in range(5)]))
say("cnn: IC aprendido por posición", ics.round(2).tolist())
say("cnn: desfase del motivo en el filtro", off)
say("cnn: consenso aprendido", "".join(BASES[i] for i in pfm_learn.argmax(1)))


def logo_tex(pfm, xs=0.46, ys=0.95, first=1, marca=None):
    Lg = len(pfm)
    t = [rf"\begin{{tikzpicture}}[x={xs}cm,y={ys}cm]"]
    if marca:
        t.append(rf"\fill[amarillo!18] ({marca[0]+0.5},0) rectangle ({marca[1]+0.5},2);")
    t.append(rf"\draw[base] (0.4,0) -- ({Lg+0.6},0);")
    t.append(r"\draw[base] (0.4,0) -- (0.4,2);")
    for v in (0, 1, 2):
        t.append(rf"\draw[base] (0.4,{v}) -- (0.3,{v}); \node[font=\sffamily\tiny,text=gris,anchor=east] at (0.3,{v}) {{{v}}};")
    t.append(r"\node[etiqueta,rotate=90] at (-0.55,1) {bits};")
    for j, p in enumerate(pfm):
        ic = ic_col(p)
        y0 = 0
        for bi in np.argsort(p):
            h = p[bi] * ic
            if h * ys > 0.025:
                b = BASES[bi]
                t.append(rf"\node[anchor=south west,inner sep=0pt,text={NUCCOL[b]}] at ({j+1-0.45:.3f},{y0:.4f}) "
                         rf"{{\resizebox{{{0.9*xs:.3f}cm}}{{{h*ys:.4f}cm}}{{\sffamily\bfseries {b}}}}};")
            y0 += h
        t.append(rf"\node[font=\sffamily\tiny,text=gris] at ({j+1},-0.16) {{{j+first}}};")
    t.append(r"\end{tikzpicture}")
    return "\n".join(t) + "\n"


pad = np.full((2, 4), 0.25)
w("logo_verdadero.tex", logo_tex(np.r_[pad, PFM, pad]))
w("logo_aprendido.tex", logo_tex(pfm_learn, marca=(off + 1, off + 7)))

# pesos crudos del filtro (heatmap 4 x 11) para la figura
Wf = net.conv.weight.detach().numpy()[fbest]  # (4, 11)
vm = np.abs(Wf).max()
t = [r"\begin{tikzpicture}[x=0.46cm,y=0.46cm]"]
for i, bse in enumerate(BASES):
    t.append(rf"\node[anchor=east,font=\ttfamily\bfseries\scriptsize,text={NUCCOL[bse]}] at (0.45,{-i}) {{{bse}}};")
    for j in range(11):
        v = Wf[i, j] / vm
        col = f"rojo!{int(90*v)}" if v > 0 else f"azulprofundo!{int(-90*v)}"
        t.append(rf"\node[draw=white,fill={col},minimum size=0.46cm,inner sep=0pt] at ({j+1},{-i}) {{}};")
for j in range(11):
    t.append(rf"\node[font=\sffamily\tiny,text=gris] at ({j+1},-3.75) {{{j+1}}};")
t.append(r"\end{tikzpicture}")
w("filtro_pesos.tex", "\n".join(t) + "\n")
say("cnn: rango de pesos del filtro", (round(float(Wf.min()), 2), round(float(Wf.max()), 2)))

# saliencia (gradiente x entrada) en una secuencia positiva del test
cons = {"TGACTCA", "TGAGTCA"}
cand = [i for i in ite if yc[i] == 1 and 30 < sims[i][1] < 60
        and sims[i][0][sims[i][1]:sims[i][1] + 7] in cons and lte[i - 5000] > 2]
i0 = cand[0]
xi = Xc[i0:i0 + 1].clone().requires_grad_(True)
net(xi).sum().backward()
gx = (xi.grad * xi).detach().numpy()[0].sum(0)
s0, at0 = sims[i0]
dat("saliencia.dat", "pos val", [(k + 1, float(v)) for k, v in enumerate(gx)])
say("sal: secuencia", s0)
say("sal: motivo en posiciones", (at0 + 1, at0 + 7, s0[at0:at0 + 7]))
say("sal: fracción |sal| dentro del motivo", round(np.abs(gx[at0:at0 + 7]).sum() / np.abs(gx).sum(), 3))
# letras escaladas por contribución en ventana alrededor del motivo
a, b = at0 - 8, at0 + 15
vmax = np.abs(gx[a:b]).max()
t = [r"\begin{tikzpicture}[x=0.5cm,y=1cm]"]
t.append(rf"\fill[amarillo!18] ({at0-a+0.5},-0.55) rectangle ({at0-a+7+0.5},1.05);")
t.append(rf"\draw[base] (0.4,0) -- ({b-a+0.6},0);")
for k in range(a, b):
    c = s0[k]; v = gx[k] / vmax
    x = k - a + 1
    if abs(v) * 0.9 > 0.03:
        if v > 0:
            t.append(rf"\node[anchor=south,inner sep=0pt,text={NUCCOL[c]}] at ({x},0) {{\resizebox{{0.42cm}}{{{v*0.9:.3f}cm}}{{\sffamily\bfseries {c}}}}};")
        else:
            t.append(rf"\node[anchor=north,inner sep=0pt,text={NUCCOL[c]}!55] at ({x},0) {{\scalebox{{1}}[-1]{{\resizebox{{0.42cm}}{{{-v*0.9:.3f}cm}}{{\sffamily\bfseries {c}}}}}}};")
    t.append(rf"\node[font=\ttfamily\tiny,text=gris] at ({x},-0.62) {{{c}}};")
t.append(rf"\node[etiqueta,anchor=west] at ({b-a+0.8},0.45) {{contribución $+$}};")
t.append(rf"\node[etiqueta,anchor=west] at ({b-a+0.8},-0.3) {{contribución $-$}};")
t.append(r"\end{tikzpicture}")
w("saliencia_letras.tex", "\n".join(t) + "\n")
w("saliencia_marca.tex", rf"\def\capdiecisieteMa{{{at0+0.5}}}\def\capdiecisieteMb{{{at0+7.5}}}" + "\n")

# =====================================================================
# 17.3  Atención y ESM-2
# =====================================================================
# ---- C1. atención de juguete ------------------------------------------
tok_toy = ["C", "A", "K", "C"]
Q = np.array([[2.0, 0.0], [0.0, 1.0], [0.5, 1.0], [2.0, 0.0]])
K = np.array([[2.0, 0.0], [0.0, 1.0], [0.0, 1.0], [2.0, 0.0]])
V = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
Sc = Q @ K.T / math.sqrt(2)
A = np.exp(Sc); A /= A.sum(1, keepdims=True)
Zt = A @ V
say("att: S = QK^T/sqrt(2)", Sc.round(3).tolist())
say("att: A", A.round(3).tolist())
say("att: Z", Zt.round(3).tolist())
t = []
for i in range(4):
    for j in range(4):
        v = A[i, j]
        col = f"azul!{int(8 + 85 * v)}"
        txtc = "white" if v > 0.45 else "tinta"
        t.append(rf"\node[draw=white,fill={col},minimum size=7.5mm,inner sep=0pt,text={txtc},font=\sffamily\scriptsize] at ({j*0.75:.2f},{-i*0.75:.2f}) {{{v:.2f}}};".replace(f"{v:.2f}", f"{v:.2f}".replace(".", ","), 1))
w("atencion_celdas.tex", "\n".join(t) + "\n")

# ---- C2-C4. ESM-2 -----------------------------------------------------
from transformers import AutoTokenizer, EsmForMaskedLM, EsmModel
from Bio.PDB import PDBParser, PPBuilder
MODEL = "facebook/esm2_t33_650M_UR50D"
tok = AutoTokenizer.from_pretrained(MODEL)
mlm = EsmForMaskedLM.from_pretrained(MODEL).eval()
AA = "RHKDESTNQCGPAVILMFYW"
aa_ids = tok.convert_tokens_to_ids(list(AA))

pdbf = fetch("https://files.rcsb.org/download/1UBQ.pdb", "1ubq.pdb")
st = PDBParser(QUIET=True).get_structure("u", str(pdbf))
chain = st[0]["A"]
res = [r for r in chain if r.id[0] == " "]
ubq = "".join(str(pp.get_sequence()) for pp in PPBuilder().build_peptides(chain))
say("esm: secuencia 1UBQ", (len(ubq), ubq))
Lu = len(ubq)

cfile = CACHE / "ubq_esm.npz"
if cfile.exists():
    zz = np.load(cfile)
    LLR, logp_all, contacts = zz["LLR"], zz["logp"], zz["contacts"]
else:
    enc = tok(ubq, return_tensors="pt")
    ids = enc["input_ids"]
    LLR = np.zeros((20, Lu)); logp_all = np.zeros((Lu, 33))
    with torch.no_grad():
        for i in range(Lu):
            m = ids.clone(); m[0, i + 1] = tok.mask_token_id
            lp = torch.log_softmax(mlm(input_ids=m).logits[0, i + 1], -1).numpy()
            logp_all[i] = lp
            wt = tok.convert_tokens_to_ids(ubq[i])
            LLR[:, i] = lp[aa_ids] - lp[wt]
        contacts = mlm.predict_contacts(ids, enc["attention_mask"])[0].numpy()
    np.savez(cfile, LLR=LLR, logp=logp_all, contacts=contacts)

i44 = 43
# MLM en I44 y en una posición tolerante
ent = [-(np.exp(logp_all[i]) * logp_all[i]).sum() for i in range(Lu)]
say("mlm: 5 posiciones de mayor entropía", [(ubq[i] + str(i + 1), round(float(ent[i]), 2)) for i in np.argsort(ent)[-5:]])
for tag, ii in (("i44", 43), ("e24", 23)):
    pp = np.exp(logp_all[ii])
    top = sorted(((pp[tok.convert_tokens_to_ids(a)], a) for a in AA), reverse=True)[:6]
    say(f"mlm: top-6 en {tag}", [(a, round(float(p), 3)) for p, a in top])
    dat(f"mlm_{tag}.dat", "k aa p", [(k, a, float(p)) for k, (p, a) in enumerate(top)])
say("mlm: residuos 40-48", ubq[39:48])
say("mlm: residuos 20-28", ubq[19:28])
# pseudo-log-verosimilitud
pll = sum(logp_all[i, tok.convert_tokens_to_ids(ubq[i])] for i in range(Lu))
say("mlm: pseudo-perplejidad", round(math.exp(-pll / Lu), 3))

meanLLR = LLR.sum(0) / 19
orden = np.argsort(meanLLR)
say("paisaje: 8 posiciones más intolerantes", [(ubq[i] + str(i + 1), round(meanLLR[i], 2)) for i in orden[:8]])
say("paisaje: 6 más tolerantes", [(ubq[i] + str(i + 1), round(meanLLR[i], 2)) for i in orden[-6:]])
say("paisaje: LLR I44V, I44A, I44D", tuple(round(LLR[AA.index(a), i44], 2) for a in "VAD"))
say("paisaje: LLR G76A, G75A, G47A", (round(LLR[AA.index('A'), 75], 2), round(LLR[AA.index('A'), 74], 2), round(LLR[AA.index('A'), 46], 2)))
say("paisaje: fracción LLR<0", round((LLR < -1e-9).sum() / (19 * Lu), 3))

# contactos verdaderos (Cbeta, Calfa para Gly) < 8 A
coords = np.array([(r["CB"] if "CB" in r else r["CA"]).coord for r in res])
Dm = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
true_c = Dm < 8.0


def prec_top(minsep, frac):
    iu = [(i, j) for i in range(Lu) for j in range(i + minsep, Lu)]
    sc = np.array([contacts[i, j] for i, j in iu])
    k = max(1, int(Lu * frac))
    sel = np.argsort(-sc)[:k]
    return np.mean([true_c[iu[s]] for s in sel])


say("contactos: prec top-L (|i-j|>=6)", round(prec_top(6, 1.0), 3))
say("contactos: prec top-L/2 (|i-j|>=12)", round(prec_top(12, 0.5), 3))
say("contactos: prec top-L/5 (|i-j|>=24)", round(prec_top(24, 0.2), 3))
say("contactos: n contactos reales |i-j|>=6", int(np.triu(true_c, 6).sum()))

# ---- figuras matplotlib ----------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.font_manager as fm
try:
    for sty in ("Regular", "Bold", "It"):
        f = subprocess.run(["kpsewhich", f"SourceSansPro-{sty}.otf"], capture_output=True, text=True).stdout.strip()
        if f:
            fm.fontManager.addfont(f)
except Exception:
    pass
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Source Sans Pro", "Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 8, "axes.edgecolor": COL["tinta2"], "axes.linewidth": 0.6,
                     "xtick.color": COL["tinta2"], "ytick.color": COL["tinta2"],
                     "axes.labelcolor": COL["tinta"], "pdf.fonttype": 42})
cdiv = LinearSegmentedColormap.from_list("cursodiv", ["#104281", "#3987E5", "#9EC5F4", "#F0EFEC",
                                                      "#F3B0AE", "#E66767", "#B8302F"])
cseq = LinearSegmentedColormap.from_list("curso", ["#FFFFFF", "#CDE2FB", "#86B6EF", "#2A78D6", "#1C5CAB", "#0D366B"])

# paisaje (rojo = perjudicial -> LLR negativo en rojo)
fig, ax = plt.subplots(figsize=(5.1, 2.15))
vmin = float(np.floor(LLR.min()))
im = ax.imshow(LLR, aspect="auto", cmap=cdiv.reversed(), norm=TwoSlopeNorm(0, vmin=vmin, vmax=max(2.0, LLR.max())),
               interpolation="nearest")
for i, c in enumerate(ubq):
    ax.plot(i, AA.index(c), marker="o", ms=1.6, color=COL["tinta"], mec="none")
ax.set_yticks(range(20)); ax.set_yticklabels(list(AA), fontsize=5.2, family="monospace")
xt = [0, 9, 19, 29, 39, 49, 59, 69, 75]
ax.set_xticks(xt); ax.set_xticklabels([str(i + 1) for i in xt], fontsize=6.5)
ax2 = ax.secondary_xaxis("top")
ax2.set_xticks(range(Lu)); ax2.set_xticklabels(list(ubq), fontsize=3.6, family="monospace")
ax2.tick_params(length=0, pad=1)
ax.set_xlabel("posición en la ubiquitina humana (1UBQ)")
ax.set_ylabel("aminoácido mutante", fontsize=7)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012, ticks=[-15, -10, -5, 0, 2])
cb.set_label("LLR (nats)", fontsize=6.5); cb.ax.tick_params(labelsize=6)
fig.savefig(OUT / "paisaje_ubq.pdf", dpi=300, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

# contactos
fig, axs = plt.subplots(1, 1, figsize=(2.9, 2.75))
ax = axs
M = np.full((Lu, Lu), np.nan)
iu = np.triu_indices(Lu, 1)
M[iu] = contacts[iu]
ax.imshow(M, cmap=cseq, vmin=0, vmax=1, interpolation="nearest", origin="upper")
ii, jj = np.where(np.tril(true_c, -6))
ax.scatter(jj, ii, s=1.6, color=COL["naranja"], marker="s", lw=0)
ax.plot([0, Lu - 1], [0, Lu - 1], color=COL["gris"], lw=0.5)
ax.set_xticks([0, 19, 39, 59, 75]); ax.set_xticklabels(["1", "20", "40", "60", "76"])
ax.set_yticks([0, 19, 39, 59, 75]); ax.set_yticklabels(["1", "20", "40", "60", "76"])
ax.text(0.0, 1.10, "triángulo superior: ESM-2 (probabilidad de contacto)", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=6.3, color=COL["azulnoche"])
ax.text(0.0, 1.03, "triángulo inferior: estructura 1UBQ (Cβ–Cβ < 8 Å)", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=6.3, color=COL["naranja"])
ax.set_xlabel("residuo $j$"); ax.set_ylabel("residuo $i$")
fig.savefig(OUT / "contactos_ubq.pdf", dpi=300, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

# ---- C4. embeddings de familias de UniProt -----------------------------
FAMS = {"globinas": "globin family", "citocromo c": "cytochrome c family",
        "peptidasa S1": "peptidase S1 family", "HSP20": "small heat shock protein (HSP20) family",
        "tiorredoxinas": "thioredoxin family"}
fam_seqs = {}
for nm, fam in FAMS.items():
    q = f'(family:"{fam}") AND (reviewed:true) AND (length:[90 TO 450])'
    url = "https://rest.uniprot.org/uniprotkb/search?format=fasta&size=500&query=" + urllib.parse.quote(q)
    txt = fetch(url, f"uniprot_{fam.replace(' ', '_').replace('(', '').replace(')', '')}.fasta").read_text()
    rec = [r for r in txt.split(">")[1:]]
    sq = ["".join(r.splitlines()[1:]) for r in rec]
    sq = [s for s in sq if set(s) <= set("ACDEFGHIKLMNPQRSTVWY")]
    order = rng.permutation(len(sq))
    chosen, vecs = [], []
    for o in order:          # reducción de redundancia por coseno de 3-mers
        v = spectrum(sq[o], 3, "ACDEFGHIKLMNPQRSTVWY")
        v /= np.linalg.norm(v)
        if all(v @ u < 0.35 for u in vecs):
            chosen.append(sq[o]); vecs.append(v)
        if len(chosen) == 30:
            break
    fam_seqs[nm] = chosen
    print(f"  familia {nm}: {len(sq)} secuencias, {len(chosen)} elegidas")

efile = CACHE / "emb_fams.npz"
names = [nm for nm in FAMS for _ in fam_seqs[nm]]
allseq = [s for nm in FAMS for s in fam_seqs[nm]]
if efile.exists() and len(np.load(efile)["E"]) == len(allseq):
    E = np.load(efile)["E"]
else:
    enc_model = mlm.esm
    E = []
    with torch.no_grad():
        for s in allseq:
            e = tok(s, return_tensors="pt")
            h = enc_model(**e).last_hidden_state[0, 1:-1]
            E.append(h.mean(0).numpy())
    E = np.array(E)
    np.savez(efile, E=E)
Kx = np.array([spectrum(s, 3, "ACDEFGHIKLMNPQRSTVWY") for s in allseq])
Kx /= np.linalg.norm(Kx, axis=1, keepdims=True)
En = E / np.linalg.norm(E, axis=1, keepdims=True)
lab = np.array([list(FAMS).index(n) for n in names])


def nn_acc(Xn):
    Sm = Xn @ Xn.T
    np.fill_diagonal(Sm, -np.inf)
    return float((lab[Sm.argmax(1)] == lab).mean())


say("emb: n secuencias", len(allseq))
say("emb: 1-NN exactitud 3-mers", round(nn_acc(Kx), 3))
say("emb: 1-NN exactitud ESM-2", round(nn_acc(En), 3))
import umap
fig, axs = plt.subplots(1, 2, figsize=(5.1, 2.45))
colf = [COL["azul"], COL["naranja"], COL["aqua"], COL["magenta"], COL["violeta"]]
for ax, Xn, tt in ((axs[0], Kx, "espectro de 3-mers"), (axs[1], En, "embedding ESM-2 (media)")):
    U = umap.UMAP(n_neighbors=12, min_dist=0.3, metric="cosine", random_state=17).fit_transform(Xn)
    for k, nm in enumerate(FAMS):
        sel = lab == k
        ax.scatter(U[sel, 0], U[sel, 1], s=9, color=colf[k], alpha=0.85, lw=0.3, ec="white", label=nm)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{tt}\n1-NN: {100*nn_acc(Xn):.0f} %", fontsize=7.5, color=COL["tinta"])
    ax.set_xlabel("UMAP 1", fontsize=6.5); ax.set_ylabel("UMAP 2", fontsize=6.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axs[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6, frameon=False, handletextpad=0.1)
fig.savefig(OUT / "embeddings.pdf", dpi=300, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

(OUT / "salida.json").write_text(json.dumps({k: str(v) for k, v in NUM.items()}, indent=1, ensure_ascii=False))
print("listo")
