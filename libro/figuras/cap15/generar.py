"""Genera datos y figuras del capítulo 15 (bioinformática estructural).

Ejecutar desde libro/:  python3 figuras/cap15/generar.py
Descarga estructuras reales del PDB (RCSB) y de AlphaFold DB a una caché
temporal, calcula ángulos, superposiciones y métricas, y escribe
.dat/.tex/.pdf en figuras/cap15/. Imprime las cifras citadas en el texto.
"""
from pathlib import Path
import json, math, tempfile, time, warnings
import numpy as np
import requests

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent
CACHE = Path(tempfile.gettempdir()) / "cap15_cache"
CACHE.mkdir(exist_ok=True)
rng = np.random.default_rng(15)
S = requests.Session()
S.headers["User-Agent"] = "bioinfo-libro-cap15 (mailto:juvenal.yosa@gmail.com)"

COL = dict(azul="#2A78D6", naranja="#EB6834", aqua="#1BAF7A", amarillo="#EDA100",
           magenta="#E87BA4", verde="#008300", violeta="#4A3AA7", rojo="#E34948",
           tinta="#0B0B0B", tinta2="#52514E", gris="#898781", rejilla="#E1E0D9",
           azulnoche="#0D366B", azulpalido="#CDE2FB", papel="#F0EFEC")


def w(name, text):
    (OUT / name).write_text(text)


def fetch(url, name):
    p = CACHE / name
    if not p.exists() or p.stat().st_size == 0:
        for k in range(4):
            try:
                r = S.get(url, timeout=60)
                r.raise_for_status()
                p.write_bytes(r.content)
                break
            except Exception:
                time.sleep(2 + 2 * k)
    return p


# ---------------------------------------------------------------------
# Utilidades: Kabsch, RMSD y TM-score
# ---------------------------------------------------------------------
def kabsch(P, Q):
    """Rotación R y traslación t que minimizan ||(P R^T + t) - Q||."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, s, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0] * (P.shape[1] - 1) + [d])
    R = Vt.T @ D @ U.T
    return R, qc - pc @ R.T


def rmsd(P, Q):
    return float(np.sqrt(((P - Q) ** 2).sum(1).mean()))


def sup_rmsd(P, Q):
    R, t = kabsch(P, Q)
    return rmsd(P @ R.T + t, Q)


def tmscore(P, Q, Lref):
    """TM-score con correspondencia fija (búsqueda heurística de semillas)."""
    d0 = max(1.24 * (Lref - 15) ** (1 / 3) - 1.8, 0.5)
    n = len(P)
    best = 0.0
    for Lf in sorted({n, n // 2, n // 4, max(n // 8, 4)}, reverse=True):
        for s0 in range(0, n - Lf + 1, max(1, Lf // 2)):
            idx = np.arange(s0, s0 + Lf)
            for _ in range(20):
                R, t = kabsch(P[idx], Q[idx])
                d = np.sqrt(((P @ R.T + t - Q) ** 2).sum(1))
                sc = (1 / (1 + (d / d0) ** 2)).sum() / Lref
                best = max(best, sc)
                new = np.where(d < d0 + 1.0)[0]
                if len(new) < 3 or np.array_equal(new, idx):
                    break
                idx = new
    return best


def ca_dict(path, chain=None, model=0):
    from Bio.PDB import MMCIFParser, PDBParser
    parser = MMCIFParser(QUIET=True) if str(path).endswith("cif") else PDBParser(QUIET=True)
    s = parser.get_structure("x", str(path))
    m = list(s)[model]
    ch = m[chain] if chain else list(m)[0]
    out = {}
    for r in ch:
        if r.id[0] == " " and "CA" in r:
            out[r.id[1]] = (r["CA"].coord.astype(float), r["CA"].get_bfactor(), r.get_resname())
    return out


# =====================================================================
# 15.1 (a) Crecimiento del PDB por método
# =====================================================================
def pdb_count(method, y):
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": [
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "exptl.method", "operator": "exact_match", "value": method}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_accession_info.initial_release_date", "operator": "range",
            "value": {"from": f"{y}-01-01", "to": f"{y}-12-31",
                      "include_lower": True, "include_upper": True}}}]},
        "return_type": "entry",
        "request_options": {"return_counts": True, "results_content_type": ["experimental"]}}
    for k in range(4):
        try:
            r = S.post("https://search.rcsb.org/rcsbsearch/v2/query", json=q, timeout=60)
            if r.status_code == 204:
                return 0
            return int(r.json()["total_count"])
        except Exception:
            time.sleep(2 + 2 * k)
    raise RuntimeError("RCSB sin respuesta")


anual = OUT / "pdb_anual.dat"
if not anual.exists():
    rows = []
    for y in range(1976, 2026):
        c = [pdb_count(m, y) for m in ("X-RAY DIFFRACTION", "SOLUTION NMR", "ELECTRON MICROSCOPY")]
        rows.append((y, *c))
    w("pdb_anual.dat", "anio xray nmr em\n" + "\n".join(" ".join(map(str, r)) for r in rows) + "\n")
A = np.loadtxt(anual, skiprows=1)
tot = A[:, 1:].sum(1)
print(f"[pdb] entradas liberadas 1976-2025: {tot.sum():,.0f}; en 2025: {tot[-1]:,.0f}")
for y in (2000, 2010, 2015, 2020, 2025):
    i = int(y - 1976)
    print(f"[pdb] {y}: rx={A[i,1]:.0f} rmn={A[i,2]:.0f} em={A[i,3]:.0f}  frac EM={A[i,3]/tot[i]:.1%}")

# =====================================================================
# 15.1 (b) Ramachandran con estructuras de alta resolución (DSSP)
# =====================================================================
from Bio.PDB import MMCIFParser, PDBList
from Bio.PDB.DSSP import DSSP

rama = OUT / "rama.npz"
if not rama.exists():
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": [
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.resolution_combined", "operator": "less_or_equal", "value": 1.0}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.selected_polymer_entity_types",
            "operator": "exact_match", "value": "Protein (only)"}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
            "operator": "range", "value": {"from": 100, "to": 400}}}]},
        "return_type": "entry", "request_options": {"paginate": {"start": 0, "rows": 2000},
                                                  "results_content_type": ["experimental"]}}
    ids = sorted(x["identifier"] for x in S.post(
        "https://search.rcsb.org/rcsbsearch/v2/query", json=q, timeout=60).json()["result_set"])
    sel = ids[:: max(1, len(ids) // 80)][:80]
    recs = []
    for pid in sel:
        p = fetch(f"https://files.rcsb.org/download/{pid}.cif", f"{pid}.cif")
        try:
            st = MMCIFParser(QUIET=True).get_structure(pid, str(p))
            dssp = DSSP(st[0], str(p), dssp="mkdssp")
        except Exception as e:
            print("  [rama] omitido", pid, str(e)[:60])
            continue
        for k in dssp.keys():
            aa, ss, phi, psi = dssp[k][1], dssp[k][2], dssp[k][4], dssp[k][5]
            if abs(phi) < 360 and abs(psi) < 360:
                recs.append((phi, psi, ss, aa))
    phi = np.array([r[0] for r in recs]); psi = np.array([r[1] for r in recs])
    ss = np.array([r[2] for r in recs]); aa = np.array([r[3] for r in recs])
    np.savez_compressed(rama, phi=phi, psi=psi, ss=ss, aa=aa, n_ent=len(set(sel)))
R = np.load(rama)
phi, psi, ss, aa = R["phi"], R["psi"], R["ss"], R["aa"]
print(f"[rama] residuos={len(phi):,}  entradas={int(R['n_ent'])}")
hel = np.isin(ss, ["H", "G", "I"]); lam = np.isin(ss, ["E", "B"])
print(f"[rama] hélice(H/G/I)={hel.mean():.1%}  lámina(E/B)={lam.mean():.1%}  otro={1-hel.mean()-lam.mean():.1%}")
H = ss == "H"; E = ss == "E"
print(f"[rama] media alfa H: phi={phi[H].mean():.1f} psi={psi[H].mean():.1f}; "
      f"lámina E: phi={phi[E].mean():.1f} psi={psi[E].mean():.1f}")
gly = aa == "G"
print(f"[rama] phi>0: global={np.mean(phi > 0):.1%}  Gly={np.mean(phi[gly] > 0):.1%}  "
      f"no-Gly={np.mean(phi[~gly] > 0):.1%}  (Gly={gly.mean():.1%} de residuos)")
pro = aa == "P"
print(f"[rama] Pro: phi medio={phi[pro].mean():.1f} sd={phi[pro].std():.1f}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import subprocess
import matplotlib.font_manager as fm
try:  # fuente sans del libro (Source Sans Pro, de TeX Live)
    for sty in ("Regular", "Bold", "It"):
        f = subprocess.run(["kpsewhich", f"SourceSansPro-{sty}.otf"], capture_output=True, text=True).stdout.strip()
        if f:
            fm.fontManager.addfont(f)
except Exception:
    pass
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Source Sans Pro", "Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 8.5, "axes.edgecolor": COL["tinta2"], "axes.linewidth": 0.6,
                     "xtick.color": COL["tinta2"], "ytick.color": COL["tinta2"],
                     "axes.labelcolor": COL["tinta"], "pdf.fonttype": 42})

fig, axs = plt.subplots(1, 2, figsize=(5.1, 2.55), sharey=True)
oth = ~(hel | lam)
for ax, mask_sets, title in [
        (axs[0], [(oth & ~gly, COL["gris"], "otros"), (lam & ~gly, COL["naranja"], "lámina β (E/B)"),
                  (hel & ~gly, COL["azul"], "hélice (H/G/I)")], "todos salvo glicina"),
        (axs[1], [(gly, COL["verde"], "glicina")], "glicina")]:
    for m, c, lab in mask_sets:
        big = lab == "glicina"
        ax.scatter(phi[m], psi[m], s=2.2 if big else 0.6, c=c, alpha=0.55 if big else 0.35,
                   lw=0, rasterized=True, label=lab)
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xticks(range(-180, 181, 90)); ax.set_yticks(range(-180, 181, 90))
    ax.axhline(0, color=COL["rejilla"], lw=0.5, zorder=0); ax.axvline(0, color=COL["rejilla"], lw=0.5, zorder=0)
    ax.set_xlabel("φ (grados)"); ax.set_title(title, fontsize=8, color=COL["tinta"])
    ax.set_aspect("equal")
axs[0].set_ylabel("ψ (grados)")
for txt, x, y in [("αR", -63, -42), ("β", -120, 135), ("αL", 60, 45), ("PPII", -70, 150)]:
    axs[0].annotate(txt, (x, y), xytext=(x + (40 if x < 0 else 35), y + (-25 if y > 100 else 25)),
                    fontsize=8, color=COL["tinta"], fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8),
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=COL["tinta2"]))
leg = axs[0].legend(loc="lower right", fontsize=6, markerscale=8, frameon=True, handletextpad=0.2,
                    borderpad=0.3, framealpha=0.9)
leg.get_frame().set_linewidth(0.3)
for lh in leg.legend_handles:
    lh.set_alpha(1)
fig.tight_layout(pad=0.3, w_pad=0.6)
fig.savefig(OUT / "ramachandran.pdf", dpi=300, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

# =====================================================================
# 15.1 (c) Ubiquitina: cristal (1UBQ) frente a ensamble de RMN (1D3Z)
# =====================================================================
from Bio.PDB import PDBParser
xr = ca_dict(fetch("https://files.rcsb.org/download/1UBQ.pdb", "1UBQ.pdb"))
nmr_path = fetch("https://files.rcsb.org/download/1D3Z.pdb", "1D3Z.pdb")
nmr_st = PDBParser(QUIET=True).get_structure("n", str(nmr_path))
res = list(range(1, 77))
X = np.array([xr[i][0] for i in res]); Bf = np.array([xr[i][1] for i in res])
models = []
for m in nmr_st:
    ch = list(m)[0]
    models.append(np.array([ch[i]["CA"].coord for i in res], float))
models = np.array(models)
core = np.arange(0, 70)
r_all = [sup_rmsd(M, X) for M in models]
r_core = []
for M in models:
    Rm, tm = kabsch(M[core], X[core]); r_core.append(rmsd((M @ Rm.T + tm)[core], X[core]))
print(f"[ubq] modelos RMN={len(models)}; RMSD Cα 1UBQ vs RMN (1-76): media={np.mean(r_all):.2f} Å "
      f"[{min(r_all):.2f}-{max(r_all):.2f}]; núcleo 1-70: {np.mean(r_core):.2f} Å")
# RMSF del ensamble superpuesto sobre el núcleo al modelo 1 (iterado a la media)
ref = models[0].copy()
for _ in range(5):
    al = []
    for M in models:
        Rm, tm = kabsch(M[core], ref[core]); al.append(M @ Rm.T + tm)
    al = np.array(al); ref = al.mean(0)
rmsf = np.sqrt(((al - ref) ** 2).sum(2).mean(0))
tm_u = tmscore(models[0], X, 76)
print(f"[ubq] TM-score modelo1 vs 1UBQ = {tm_u:.3f};  RMSF máx en res {res[int(rmsf.argmax())]} = {rmsf.max():.2f} Å")
print(f"[ubq] B medio 1-70={Bf[:70].mean():.1f}  B 72-76={Bf[71:].mean():.1f} Å^2")
print(f"[ubq] corr(B, RMSF) = {np.corrcoef(Bf, rmsf)[0,1]:.2f}")
w("ubq.dat", "res B rmsf\n" + "\n".join(f"{r} {b:.2f} {f:.3f}" for r, b, f in zip(res, Bf, rmsf)) + "\n")
# u = sqrt(3B/8pi^2)
for b in (10, 20, 40, 80):
    print(f"[bfac] B={b}: desplazamiento rms 3D = {math.sqrt(3*b/(8*math.pi**2)):.2f} Å")

# =====================================================================
# 15.1 (d) Mioglobina (1A6M) frente a hemoglobina α (2DN2:A)
# =====================================================================
from Bio import Align
from Bio.Align import substitution_matrices
from Bio.SeqUtils import seq1
mb = ca_dict(fetch("https://files.rcsb.org/download/1A6M.pdb", "1A6M.pdb"), "A")
hb = ca_dict(fetch("https://files.rcsb.org/download/2DN2.pdb", "2DN2.pdb"), "A")
mk, hk = sorted(mb), sorted(hb)
smb = "".join(seq1(mb[k][2]) for k in mk); shb = "".join(seq1(hb[k][2]) for k in hk)
al = Align.PairwiseAligner(mode="global", substitution_matrix=substitution_matrices.load("BLOSUM62"),
                           open_gap_score=-10, extend_gap_score=-0.5)
aln = al.align(smb, shb)[0]
pairs = [(i, j) for (a0, a1), (b0, b1) in zip(*aln.aligned) for i, j in zip(range(a0, a1), range(b0, b1))]
ident = sum(smb[i] == shb[j] for i, j in pairs)
P = np.array([mb[mk[i]][0] for i, _ in pairs]); Q = np.array([hb[hk[j]][0] for _, j in pairs])
r_mh = sup_rmsd(P, Q); tm_mh = tmscore(P, Q, len(shb))
print(f"[mb-hb] Lmb={len(smb)} Lhb={len(shb)} pares={len(pairs)} identidad={ident/len(pairs):.1%} "
      f"RMSD={r_mh:.2f} Å  TM(norm Hbα)={tm_mh:.3f}")
# RMSD de núcleo: descarta iterativamente pares con d > 3 Å
idx = np.arange(len(P))
for _ in range(10):
    Rm, tm = kabsch(P[idx], Q[idx]); d = np.sqrt(((P @ Rm.T + tm - Q) ** 2).sum(1))
    new = np.where(d < 3.0)[0]
    if np.array_equal(new, idx): break
    idx = new
print(f"[mb-hb] núcleo d<3 Å: {len(idx)} pares, RMSD={rmsd((P @ Rm.T + tm)[idx], Q[idx]):.2f} Å")
# Pares aleatorios de estructuras no relacionadas: TM de referencia
Pr = P.copy(); rng.shuffle(Pr)
print(f"[tm] TM de correspondencia aleatoria (mb barajada) = {tmscore(Pr, Q, len(shb)):.3f}")

# =====================================================================
# 15.1 (e) Ejemplo numérico de Kabsch en 2D (figura esquemática)
# =====================================================================
Q2 = np.array([[0.0, 0.0], [1.5, 0.3], [2.4, 1.6], [1.2, 2.5], [-0.3, 1.4]])
th = np.deg2rad(52.0)
Rt = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
noise = np.array([[0.08, -0.05], [-0.06, 0.07], [0.05, 0.04], [-0.07, -0.06], [0.04, 0.06]])
P2 = (Q2 - Q2.mean(0)) @ Rt.T + np.array([5.2, 0.4]) + noise
R2, t2 = kabsch(P2, Q2)
P2a = P2 @ R2.T + t2
ang = math.degrees(math.atan2(R2[1, 0], R2[0, 0]))
print(f"[kabsch2d] RMSD antes={rmsd(P2, Q2):.2f}  tras centrar={rmsd(P2-P2.mean(0), Q2-Q2.mean(0)):.2f}"
      f"  tras Kabsch={rmsd(P2a, Q2):.3f}  ángulo={ang:.1f}°")
Hc = (P2 - P2.mean(0)).T @ (Q2 - Q2.mean(0))
U, sv, Vt = np.linalg.svd(Hc)
print(f"[kabsch2d] H={np.round(Hc,3).tolist()} sv={np.round(sv,3).tolist()} det={np.linalg.det(Vt.T@U.T):.0f}")
def pts(M, name):
    return "".join(f"\\coordinate ({name}{i}) at ({x:.3f},{y:.3f});\n" for i, (x, y) in enumerate(M))
w("kabsch_pts.tex", pts(Q2, "q") + pts(P2, "p") + pts(P2 - P2.mean(0) + Q2.mean(0), "c") + pts(P2a, "a")
  + f"\\def\\capquinceRantes{{{rmsd(P2, Q2):.2f}}}\\def\\capquinceRcentro{{{rmsd(P2-P2.mean(0), Q2-Q2.mean(0)):.2f}}}"
  + f"\\def\\capquinceRfinal{{{rmsd(P2a, Q2):.2f}}}\\def\\capquinceAng{{{-ang:.0f}}}\n")

# =====================================================================
# 15.2 AlphaFold DB: p53 humana (P04637) — pLDDT y PAE
# =====================================================================
af = fetch("https://alphafold.ebi.ac.uk/files/AF-P04637-F1-model_v6.pdb", "AF-P04637.pdb")
pae_p = fetch("https://alphafold.ebi.ac.uk/files/AF-P04637-F1-predicted_aligned_error_v6.json", "AF-P04637-pae.json")
afd = ca_dict(af)
ak = sorted(afd)
pl = np.array([afd[k][1] for k in ak])
w("plddt.dat", "res plddt\n" + "\n".join(f"{k} {v:.2f}" for k, v in zip(ak, pl)) + "\n")
bands = [(90, 101, "muy alta"), (70, 90, "segura"), (50, 70, "baja"), (0, 50, "muy baja")]
for lo, hi, nm in bands:
    print(f"[plddt] {nm}: {np.mean((pl >= lo) & (pl < hi)):.1%}")
for a, b, nm in [(1, 93, "N-term 1-93"), (94, 292, "DBD 94-292"), (293, 324, "enlace"), (325, 356, "tetram 325-356"), (357, 393, "C-term 357-393")]:
    m = (np.array(ak) >= a) & (np.array(ak) <= b)
    print(f"[plddt] {nm}: media={pl[m].mean():.1f}")
pj = json.loads(pae_p.read_text())
pj = pj[0] if isinstance(pj, list) else pj
PAE = np.array(pj["predicted_aligned_error"], float)
print(f"[pae] forma={PAE.shape} máx={PAE.max():.1f}")
dbd = slice(93, 292); tet = slice(324, 356)
print(f"[pae] DBD-DBD media={PAE[dbd, dbd].mean():.1f}  tet-tet={PAE[tet, tet].mean():.1f}  "
      f"DBD->tet={PAE[dbd, tet].mean():.1f}  Nterm-Nterm={PAE[:60,:60].mean():.1f}")
cm = LinearSegmentedColormap.from_list("cursopae", [COL["azulnoche"], COL["azul"], "#9EC5F4", "#F7F7F4"])
fig, ax = plt.subplots(figsize=(3.3, 2.8))
im = ax.imshow(PAE, cmap=cm, vmin=0, vmax=30, origin="upper", interpolation="nearest",
               extent=(0.5, PAE.shape[1] + 0.5, PAE.shape[0] + 0.5, 0.5))
for a, b, nm in [(94, 292, "DBD"), (325, 356, "TET")]:
    ax.add_patch(plt.Rectangle((a, a), b - a, b - a, fill=False, ec=COL["naranja"], lw=0.9))
    ax.text(b + 6, a + 8, nm, color=COL["naranja"], fontsize=7, fontweight="bold", va="top")
ax.set_xlabel("residuo alineado $j$"); ax.set_ylabel("residuo de referencia $i$")
ax.set_xticks([1, 100, 200, 300, 393]); ax.set_yticks([1, 100, 200, 300, 393])
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("PAE esperado (Å)"); cb.outline.set_linewidth(0.4)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "pae_p53.pdf", dpi=300, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

# Estructura experimental del dominio de unión a ADN (2OCJ:A) frente al modelo
xp = ca_dict(fetch("https://files.rcsb.org/download/2OCJ.pdb", "2OCJ.pdb"), "A")
com = [k for k in sorted(xp) if k in afd and 94 <= k <= 292]
P = np.array([afd[k][0] for k in com]); Q = np.array([xp[k][0] for k in com])
print(f"[p53] DBD 2OCJ:A vs AF: {len(com)} Cα  RMSD={sup_rmsd(P, Q):.2f} Å  TM={tmscore(P, Q, len(com)):.3f}")

# =====================================================================
# 15.2 Levinthal
# =====================================================================
n, k, tau = 100, 3, 1e-13
tot_s = k ** n * tau
print(f"[levinthal] {k}^{n} = {k**n:.2e} conformaciones; a {tau:g} s cada una: {tot_s:.2e} s "
      f"= {tot_s/3.156e7:.2e} años (edad del universo ~1.4e10)")

# =====================================================================
# 15.3 Docking: términos de Vina y embudo de poses simuladas
# =====================================================================
def vina_terms(dsurf):
    g1 = np.exp(-(dsurf / 0.5) ** 2)
    g2 = np.exp(-((dsurf - 3.0) / 2.0) ** 2)
    rep = np.where(dsurf < 0, dsurf ** 2, 0.0)
    hyd = np.clip((1.5 - dsurf) / 1.0, 0, 1)
    hb = np.clip(-dsurf / 0.7, 0, 1)
    return g1, g2, rep, hyd, hb
W = dict(g1=-0.0356, g2=-0.00516, rep=0.840, hyd=-0.0351, hb=-0.587)
r = np.linspace(2.4, 8.0, 400)
ds = r - 1.9 - 1.9                                  # dos carbonos: R_C = 1,9 Å
g1, g2, rep, hyd, hb = vina_terms(ds)
e_cc = W["g1"] * g1 + W["g2"] * g2 + W["rep"] * rep + W["hyd"] * hyd
print(f"[vina] C···C mínimo en r={r[e_cc.argmin()]:.2f} Å, e={e_cc.min():.3f}")
ds2 = r - 1.8 - 1.7                                 # N donador ··· O aceptor
g1b, g2b, repb, _, hbb = vina_terms(ds2)
e_hb = W["g1"] * g1b + W["g2"] * g2b + W["rep"] * repb + W["hb"] * hbb
print(f"[vina] N···O puente H mínimo en r={r[e_hb.argmin()]:.2f} Å, e={e_hb.min():.3f}")
w("vina.dat", "r ecc ehb\n" + "\n".join(f"{a:.4f} {b:.5f} {c:.5f}" for a, b, c in zip(r, e_cc, e_hb)) + "\n")
for nrot in (0, 5, 10):
    print(f"[vina] factor 1/(1+0.0585*Nrot) con Nrot={nrot}: {1/(1+0.0585*nrot):.3f}")

# Embudo: 400 poses simuladas (puntuación frente a RMSD a la pose nativa)
npose = 400
rm = np.concatenate([rng.uniform(0.3, 12, npose - 60), rng.normal(7.5, 0.6, 60).clip(5.5, 9.5)])
sc = -9.2 + 4.8 * (1 - np.exp(-rm / 3.0)) + rng.normal(0, 0.55, npose)
trap = np.arange(npose - 60, npose)
sc[trap] = -7.9 + 0.35 * (rm[trap] - 7.5) ** 2 + rng.normal(0, 0.4, 60)
w("poses.dat", "rmsd score\n" + "\n".join(f"{a:.3f} {b:.3f}" for a, b in zip(rm, sc)) + "\n")
best = np.argsort(sc)[:10]
print(f"[embudo] top-1: RMSD={rm[best[0]]:.2f} score={sc[best[0]]:.2f}; top-10 con RMSD<2: "
      f"{np.sum(rm[best] < 2)}; éxito top-1={rm[best[0]] < 2}")

# Paisaje 2D (coordenadas de pose) para el esquema de embudo
xg = np.linspace(-3, 3, 61)
XX, YY = np.meshgrid(xg, xg)
Z = -9 * np.exp(-((XX - 0.6) ** 2 + (YY - 0.4) ** 2) / 1.6) - 6.5 * np.exp(-((XX + 1.7) ** 2 + (YY + 1.4) ** 2) / 0.5) \
    + 0.35 * np.sin(3 * XX) * np.cos(3 * YY)
w("paisaje.dat", "\n\n".join("\n".join(f"{XX[i,j]:.2f} {YY[i,j]:.2f} {Z[i,j]:.3f}" for j in range(61))
                                   for i in range(61)) + "\n")
print("[ok] figuras/cap15 actualizadas")
