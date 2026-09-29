"""Genera datos de figuras y cifras de los ejemplos del capítulo 9
(Detección de variantes).

Ejecutar desde libro/:  python3 figuras/cap09/generar.py
Escribe .dat en figuras/cap09/ e imprime las cifras que se citan en el
texto (única fuente de verdad).
"""
from pathlib import Path
import math
import numpy as np
from scipy import stats

OUT = Path(__file__).parent
rng = np.random.default_rng(909)


def w(name, text):
    (OUT / name).write_text(text)


# =====================================================================
# 9.1  Modelo de verosimilitud de genotipos (diploide, 4 alelos)
# =====================================================================
def p_base(b, a, eps):
    """P(base observada b | alelo verdadero a) con error eps repartido
    uniformemente entre las otras tres bases."""
    return 1 - eps if b == a else eps / 3


def lik(reads, g):
    """P(D | G) para genotipo g=(a1, a2); reads = [(base, Q), ...]."""
    L = 1.0
    for b, q in reads:
        e = 10 ** (-q / 10)
        L *= 0.5 * p_base(b, g[0], e) + 0.5 * p_base(b, g[1], e)
    return L


def log10lik(reads, g):
    s = 0.0
    for b, q in reads:
        e = 10 ** (-q / 10)
        s += math.log10(0.5 * p_base(b, g[0], e) + 0.5 * p_base(b, g[1], e))
    return s


def priors(theta):
    return {"RR": 1 - 1.5 * theta, "RA": theta, "AA": theta / 2}


R, A = "A", "G"
GENO = {"RR": (R, R), "RA": (R, A), "AA": (A, A)}

reads = [("A", 30), ("A", 30), ("A", 20), ("G", 30), ("G", 30), ("G", 10)]
print("[ej] lecturas:", reads)
for b, q in reads:
    e = 10 ** (-q / 10)
    print(f"[ej]  {b} Q{q}: e={e:g}  P|RR={0.5*p_base(b,R,e)+0.5*p_base(b,R,e):.6f}"
          f"  P|RA={0.5*p_base(b,R,e)+0.5*p_base(b,A,e):.6f}"
          f"  P|AA={0.5*p_base(b,A,e)+0.5*p_base(b,A,e):.6f}")
ll = {k: log10lik(reads, g) for k, g in GENO.items()}
m = max(ll.values())
PL = {k: round(-10 * (v - m)) for k, v in ll.items()}
for k in GENO:
    print(f"[ej] L({k}) = {10**ll[k]:.4e}   log10 = {ll[k]:.3f}"
          f"   PL(no redondeado) = {-10*(ll[k]-m):.2f}  PL = {PL[k]}")
for theta in (1e-3,):
    pr = priors(theta)
    num = {k: 10 ** ll[k] * pr[k] for k in GENO}
    Z = sum(num.values())
    post = {k: num[k] / Z for k in GENO}
    best = max(post, key=post.get)
    gq = -10 * math.log10(1 - post[best])
    qual = -10 * math.log10(post["RR"])
    print(f"[ej] theta={theta}: priors={pr}")
    print("[ej] posterior:", {k: f"{v:.6g}" for k, v in post.items()},
          f" GQ={gq:.2f}  QUAL={qual:.2f}")
    # sin prior (uniforme)
    Zu = sum(10 ** ll[k] for k in GENO)
    print("[ej] posterior uniforme:",
          {k: f"{10**ll[k]/Zu:.6g}" for k in GENO})
    # GQ como diferencia de PL
    s = sorted(-10 * (v - m) for v in ll.values())
    print(f"[ej] GQ (2º menor PL - menor) = {s[1]-s[0]:.2f}")

# Ejemplo de baja profundidad: 2 lecturas, ambas A... y 1 G de Q30
for rr in ([("G", 30)], [("A", 30), ("G", 30)], [("G", 30), ("G", 30)]):
    ll2 = {k: log10lik(rr, g) for k, g in GENO.items()}
    pr = priors(1e-3)
    num = {k: 10 ** ll2[k] * pr[k] for k in GENO}
    Z = sum(num.values())
    print("[baja]", rr, {k: f"{num[k]/Z:.4f}" for k in GENO},
          " PL:", {k: round(-10 * (v - max(ll2.values()))) for k, v in ll2.items()})

# ---------- Figura: log-verosimilitudes al acumular lecturas ----------
nmax = 40
q = 30
e = 10 ** (-q / 10)
seq_reads = []
for i in range(nmax):
    allele = R if rng.random() < 0.5 else A
    if rng.random() < e:
        allele = rng.choice([x for x in "ACGT" if x != allele])
    seq_reads.append((str(allele), q))
rows = ["n kalt RR RA AA"]
for n in range(1, nmax + 1):
    rr = seq_reads[:n]
    l = {k: log10lik(rr, g) for k, g in GENO.items()}
    mx = max(l.values())
    kalt = sum(b == A for b, _ in rr)
    rows.append(f"{n} {kalt} {l['RR']-mx:.4f} {l['RA']-mx:.4f} {l['AA']-mx:.4f}")
w("lik_acum.dat", "\n".join(rows))
print("[acum] bases:", "".join(b for b, _ in seq_reads))
print("[acum] n=10:", rows[10], "  n=40:", rows[40])


# ---------- Figura: probabilidad de llamar bien un heterocigoto ----------
def p_call_het(n, theta, q=30):
    """Promedia sobre k ~ Bin(n, p_alt) (p_alt con error) la indicadora
    de que el genotipo de máxima posterior sea RA."""
    e = 10 ** (-q / 10)
    palt = 0.5 * (1 - e) + 0.5 * e / 3
    pr = priors(theta) if theta else {"RR": 1, "RA": 1, "AA": 1}
    tot = 0.0
    for k in range(n + 1):
        rr = [(A, q)] * k + [(R, q)] * (n - k)
        l = {g: log10lik(rr, GENO[g]) + math.log10(pr[g]) for g in GENO}
        if max(l, key=l.get) == "RA":
            tot += stats.binom.pmf(k, n, palt)
    return tot


rows = ["n uniforme theta3 theta4"]
for n in range(1, 41):
    rows.append(f"{n} {p_call_het(n, 0):.5f} {p_call_het(n, 1e-3):.5f}"
                f" {p_call_het(n, 1e-4):.5f}")
w("p_het.dat", "\n".join(rows))
for n in (4, 6, 8, 10, 15, 20, 30):
    print(f"[phet] n={n}: uniforme={p_call_het(n,0):.4f}"
          f" theta=1e-3: {p_call_het(n,1e-3):.4f}"
          f" theta=1e-4: {p_call_het(n,1e-4):.4f}")
# 1 - P(k=0 o k=n) sin errores (cota)
for n in (4, 6, 10):
    print(f"[phet] prob. de no ver ambos alelos n={n}: {2*0.5**n:.4f}")

# ---------- Figura: posterior en función de k (n = 20) ----------
n = 20
rows = ["k RR RA AA"]
for k in range(n + 1):
    rr = [(A, 30)] * k + [(R, 30)] * (n - k)
    pr = priors(1e-3)
    num = {g: 10 ** log10lik(rr, GENO[g]) * pr[g] for g in GENO}
    Z = sum(num.values())
    rows.append(f"{k} {num['RR']/Z:.6f} {num['RA']/Z:.6f} {num['AA']/Z:.6f}")
    if k in (0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20):
        print(f"[postk] k={k}: RR={num['RR']/Z:.4g} RA={num['RA']/Z:.4g}"
              f" AA={num['AA']/Z:.4g}")
w("post_k.dat", "\n".join(rows))

# ---------- Llamada conjunta: EM de la frecuencia alélica (Li 2011) ----------
# Modelo bialélico: P(b|G) con g copias alternativas de 2.
muestras = {
    "M1": (1, 3), "M2": (2, 4), "M3": (0, 3), "M4": (3, 5), "M5": (1, 2),
    "M6": (0, 4), "M7": (2, 3), "M8": (0, 2),
}  # (lecturas alt, profundidad), todas Q20
qj = 20
ej = 10 ** (-qj / 10)


def lik_bial(k, n, g, e):
    """P(D|g) bialélico: cada lectura muestra alt con prob
    f_g = g/2 (1-e) + (1-g/2) e (Li 2011: error hacia el otro alelo)."""
    f = g / 2 * (1 - e) + (1 - g / 2) * e
    return f ** k * (1 - f) ** (n - k)


Lg = {s: np.array([lik_bial(k, nn, g, ej) for g in (0, 1, 2)])
      for s, (k, nn) in muestras.items()}
psi = 0.5
hist = []
for it in range(50):
    pri = np.array([(1 - psi) ** 2, 2 * psi * (1 - psi), psi ** 2])
    acc = 0.0
    for s in muestras:
        post = Lg[s] * pri / (Lg[s] * pri).sum()
        acc += (post * np.array([0, 1, 2])).sum()
    new = acc / (2 * len(muestras))
    hist.append(new)
    if abs(new - psi) < 1e-10:
        psi = new
        break
    psi = new
print(f"[em] iteraciones={len(hist)}  psi={psi:.4f}  primeras={[round(h,4) for h in hist[:4]]}")
pri = np.array([(1 - psi) ** 2, 2 * psi * (1 - psi), psi ** 2])
for s in ("M1", "M5", "M3", "M8"):
    post = Lg[s] * pri / (Lg[s] * pri).sum()
    pr1 = np.array([1 - 1.5e-3, 1e-3, 0.5e-3])
    post1 = Lg[s] * pr1 / (Lg[s] * pr1).sum()
    print(f"[em] {s} {muestras[s]}: conjunto RR/RA/AA={np.round(post,4)}"
          f"   individual(theta=1e-3)={np.round(post1,4)}")

# =====================================================================
# 9.2  ti/tv, normalización, curva precisión-sensibilidad
# =====================================================================
RT, Robs = 2.1, 1.8
pT, pobs, pF = RT / (1 + RT), Robs / (1 + Robs), 1 / 3
alpha = (pT - pobs) / (pT - pF)
print(f"[titv] pT={pT:.4f} pobs={pobs:.4f} pF={pF:.4f} alfa={alpha:.4f}")
Robs2 = 2.0
pobs2 = Robs2 / (1 + Robs2)
print(f"[titv] Robs=2.0 -> alfa={(pT-pobs2)/(pT-pF):.4f}")


def normalizar(pos, ref, alt, genoma):
    """Algoritmo de Tan et al. (2015) para un par REF/ALT (pos 1-based).
    genoma: cadena de la referencia completa."""
    while True:
        cambio = False
        if ref and alt and ref[-1] == alt[-1]:
            ref, alt = ref[:-1], alt[:-1]
            cambio = True
        if not ref or not alt:
            pos -= 1
            b = genoma[pos - 1]
            ref, alt = b + ref, b + alt
            cambio = True
        if not cambio:
            break
    while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
        ref, alt, pos = ref[1:], alt[1:], pos + 1
    return pos, ref, alt


G = "GGATCACACACAGT"
# deleción de un "CA" en el bloque (CA)x4 : posiciones 5-12
print("[norm] genoma:", G, " bloque CA en 5..12")
reps = [(10, "ACA", "A"), (9, "CAC", "C"), (6, "ACA", "A"),
        (8, "ACACA", "ACA"), (4, "TCACA", "TCA"), (11, "CAG", "G")]
for r in reps:
    p, rf, al = r
    assert G[p - 1:p - 1 + len(rf)] == rf, r
    alt_seq = G[:p - 1] + al + G[p - 1 + len(rf):]
    assert alt_seq == G[:4] + G[6:], r
    print(f"[norm] {r} -> hap={alt_seq} -> {normalizar(p, rf, al, G)}")
hap = G[:4] + G[6:]
print("[norm] haplotipo alternativo:", hap)

# ---------- Curva precisión-sensibilidad por umbral de QUAL ----------
nT_verdad = 10000           # variantes en el conjunto de verdad
detect = 0.985               # fracción que el llamador llega a proponer
nTP = int(nT_verdad * detect)
nFP = 1500
qual_T = np.clip(rng.lognormal(np.log(180), 0.75, nTP), 3, 999)
qual_F = np.clip(rng.lognormal(np.log(18), 0.9, nFP), 3, 999)
rows = ["t prec sens f1"]
best = (0, None)
for t in np.concatenate([np.arange(0, 100, 1), np.arange(100, 401, 5)]):
    tp = (qual_T >= t).sum()
    fp = (qual_F >= t).sum()
    fn = nT_verdad - tp
    prec = tp / (tp + fp) if tp + fp else 1.0
    sens = tp / (tp + fn)
    f1 = 2 * prec * sens / (prec + sens)
    rows.append(f"{t:.0f} {prec:.5f} {sens:.5f} {f1:.5f}")
    if f1 > best[0]:
        best = (f1, t, prec, sens, tp, fp, fn)
w("pr_qual.dat", "\n".join(rows))
print("[pr] mejor F1=%.4f en t=%d  prec=%.4f sens=%.4f TP=%d FP=%d FN=%d" % best)
for t in (0, 20, 30, 50, 100):
    tp = (qual_T >= t).sum(); fp = (qual_F >= t).sum(); fn = nT_verdad - tp
    print(f"[pr] t={t}: TP={tp} FP={fp} FN={fn} prec={tp/(tp+fp):.4f}"
          f" sens={tp/nT_verdad:.4f} F1={2*tp/(2*tp+fp+fn):.4f}")

# Ejemplo hap.py (cifras ilustrativas redondas)
TP, FP, FN = 3_290_000, 9_800, 21_500
prec = TP / (TP + FP)
sens = TP / (TP + FN)
print(f"[happy] prec={prec:.5f} sens={sens:.5f} F1={2*prec*sens/(prec+sens):.5f}")

# =====================================================================
# 9.3  Consecuencias de todas las SNV posibles en el código estándar
# =====================================================================
from Bio.Data import CodonTable
tabla = CodonTable.unambiguous_dna_by_id[1]
fwd, stops = tabla.forward_table, set(tabla.stop_codons)
aa = lambda c: "*" if c in stops else fwd[c]
cont = {p: {"sin": 0, "mis": 0, "non": 0, "stoplost": 0, "stopret": 0}
        for p in (1, 2, 3)}
for c in [a + b + d for a in "TCAG" for b in "TCAG" for d in "TCAG"]:
    for p in range(3):
        for nb in "ACGT":
            if nb == c[p]:
                continue
            c2 = c[:p] + nb + c[p + 1:]
            a1, a2 = aa(c), aa(c2)
            if a1 == "*":
                k = "stopret" if a2 == "*" else "stoplost"
            elif a2 == "*":
                k = "non"
            elif a1 == a2:
                k = "sin"
            else:
                k = "mis"
            cont[p + 1][k] += 1
rows = ["pos sin mis non stoplost stopret"]
for p in (1, 2, 3):
    c = cont[p]
    rows.append(f"{p} {c['sin']} {c['mis']} {c['non']} {c['stoplost']} {c['stopret']}")
    print(f"[cod] posición {p}: {c}")
w("snv_codon.dat", "\n".join(rows))
tot = {k: sum(cont[p][k] for p in (1, 2, 3)) for k in cont[1]}
sense = 61 * 9
print(f"[cod] total: {tot}  (sobre codones sentido: {sense})")
print(f"[cod] fracción sinónima en codones sentido: {tot['sin']/sense:.4f}"
      f"  missense: {tot['mis']/sense:.4f}  nonsense: {tot['non']/sense:.4f}")

# CADD: escala Phred del rango
for C in (10, 20, 30):
    print(f"[cadd] C={C}: top {100*10**(-C/10):g}% ")
