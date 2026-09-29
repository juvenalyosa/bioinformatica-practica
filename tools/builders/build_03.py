from nbbuild import NB, SETUP, header, gif

PATH = "modulo-00-preparacion/0.3_reproducibilidad.ipynb"
nb = NB()

nb.md(header(PATH, "Lección 0.3 · Reproducibilidad: semillas, versiones, huellas digitales y Git",
             "Módulo 0 — Preparación del laboratorio digital",
             "~2.5 horas", "Introductorio–intermedio", "Lecciones 0.1 y 0.2"))

nb.md(r"""
## 🎯 Objetivos de aprendizaje

1. **Explicar** qué es un número pseudoaleatorio y por qué una **semilla** hace reproducible una simulación.
2. **Detectar** un mal generador aleatorio (RANDU) con una visualización.
3. **Cuantificar** la incertidumbre de una simulación Monte Carlo con el error estándar $\sigma/\sqrt{N}$.
4. **Registrar** el entorno de cómputo (versiones) y **verificar** la integridad de archivos con *hashes*.
5. **Usar** Git para versionar un análisis y conectarlo con GitHub y Colab.

## 🗺️ Mapa de la clase

1. ¿Por qué la reproducibilidad es un problema?
2. El azar de la computadora: números pseudoaleatorios y semillas
3. 🧪 Experimento: un generador defectuoso (RANDU)
4. 🧪 Experimento Monte Carlo: lecturas de secuenciación sin errores
5. Registrar el entorno: versiones de paquetes
6. 🧪 Huellas digitales de archivos: *hashes*
7. Git y GitHub: la máquina del tiempo del análisis
8. Lista de verificación, ejercicios y resumen
""")

nb.md(r"""
## 1. ¿Por qué la reproducibilidad es un problema?

Una receta que dice "agregue *un poco* de sal y hornee *hasta que esté listo*" funciona para quien la escribió,
pero nadie más obtiene el mismo pastel. Un análisis bioinformático sin semillas, sin versiones de software y sin
registro exacto de los datos es esa receta: funcionó una vez, en una computadora, un día, y nadie — ni siquiera su
autor seis meses después — puede garantizar que obtendrá lo mismo.

Esto no es una preocupación teórica. En 2016 una encuesta de *Nature* a 1 576 investigadores encontró que más del
70 % había intentado sin éxito reproducir el experimento de otro grupo, y más de la mitad no había logrado
reproducir **el suyo propio**. En bioinformática tenemos una ventaja enorme sobre el laboratorio húmedo: la
computadora es determinista, así que si registramos **todo** lo que influye en el resultado, la reproducción puede
ser exacta, bit por bit.

Un resultado computacional es **reproducible** cuando otra persona, con los **mismos datos** y el **mismo código**,
obtiene **el mismo resultado**. Parece trivial, pero fallan cosas como:

| Fuente de variación | Ejemplo | Solución en esta clase |
|---|---|---|
| Aleatoriedad | *bootstrap*, t-SNE/UMAP, simulaciones | **Semillas** (§2–4) |
| Software | `pandas` 1.x vs 2.x cambia un resultado por defecto | **Registrar versiones** (§5) |
| Datos | archivo corrupto o reemplazado por otra versión | **Hashes** (§6) |
| Código | "¿qué versión del script generó la Figura 3?" | **Git** (§7) |
""")

nb.code(SETUP)

nb.md(r"""
## 2. El azar de la computadora: números pseudoaleatorios

Una computadora es una máquina **determinista**: no puede lanzar una moneda. Lo que hace es calcular una sucesión de
números que **parecen** aleatorios: números **pseudoaleatorios**.

Piense en una **baraja gigantesca que alguien barajó de antemano** y dejó sobre la mesa. Cada vez que usted pide un
número aleatorio, la computadora voltea la siguiente carta. Para quien no conoce el orden, las cartas parecen
completamente al azar; pero el orden está fijo. La **semilla** (*seed*) dice **en qué carta empezar**: misma semilla
→ mismas cartas → mismos resultados. Sin semilla, la computadora elige un punto de partida distinto cada vez (por
ejemplo, a partir del reloj), y cada ejecución da resultados diferentes.

El generador más sencillo de entender es el **generador congruencial lineal** (LCG):

$$
X_{n+1} \;=\; \big(\,a \cdot X_n \;+\; c\,\big) \bmod m,
\qquad\qquad
U_n \;=\; \frac{X_n}{m} \in [0, 1)
$$

| Símbolo | Significado |
|---|---|
| $X_0$ | la **semilla**: el punto de partida |
| $X_n$ | el $n$-ésimo número entero de la sucesión |
| $a$ | multiplicador |
| $c$ | incremento |
| $m$ | módulo: $X_n$ siempre queda entre $0$ y $m-1$ |
| $\bmod$ | "residuo de la división" (como las horas de un reloj: $13 \bmod 12 = 1$) |
| $U_n$ | el número "aleatorio" final, escalado al intervalo $[0,1)$ |

**Ejemplo a mano con números pequeños.** Tome $a = 5$, $c = 1$, $m = 16$ y semilla $X_0 = 0$:

| $n$ | cálculo | $X_n$ | $U_n = X_n / 16$ |
|---|---|---|---|
| 1 | $(5 \cdot 0 + 1) \bmod 16$ | 1 | 0.0625 |
| 2 | $(5 \cdot 1 + 1) \bmod 16$ | 6 | 0.375 |
| 3 | $(5 \cdot 6 + 1) \bmod 16 = 31 \bmod 16$ | 15 | 0.9375 |
| 4 | $(5 \cdot 15 + 1) \bmod 16 = 76 \bmod 16$ | 12 | 0.75 |

La sucesión 1, 6, 15, 12, … parece desordenada, pero es completamente predecible. Y como sólo hay 16 valores
posibles, tarde o temprano se **repite**: después de, a lo sumo, 16 pasos vuelve al comienzo (el **período**). Los
generadores reales usan $m$ enormes ($2^{64}$ o más) para que el período sea astronómico.

Implementémoslo en cuatro líneas:
""")

nb.code(r'''
def lcg(seed: int, n: int, a: int = 1_103_515_245, c: int = 12_345, m: int = 2**31) -> np.ndarray:
    """Generador congruencial lineal (parámetros clásicos de la biblioteca de C)."""
    x, out = seed, np.empty(n)
    for i in range(n):
        x = (a * x + c) % m
        out[i] = x / m
    return out

print("semilla 42:", np.round(lcg(42, 5), 4))
print("semilla 42:", np.round(lcg(42, 5), 4), " ← ¡idénticos!")
print("semilla 7 :", np.round(lcg(7, 5), 4), " ← otra sucesión")
''')

nb.md(r"""
### Semillas en la práctica con `numpy`

En código moderno **no** use `np.random.seed()` (estado global compartido por todo el programa). Cree un
**generador propio** con `np.random.default_rng(semilla)` y páselo a sus funciones. Internamente usa **PCG64**, un
generador mucho mejor que el LCG.
""")

nb.code(r'''
rng_a = np.random.default_rng(2024)
rng_b = np.random.default_rng(2024)
rng_c = np.random.default_rng()            # sin semilla: distinto cada vez que ejecute la celda

bases = np.array(list("ACGT"))
print("rng_a:", "".join(rng_a.choice(bases, 30)))
print("rng_b:", "".join(rng_b.choice(bases, 30)), "← misma semilla, misma secuencia")
print("rng_c:", "".join(rng_c.choice(bases, 30)), "← ejecute la celda de nuevo: cambia")
''')

nb.md(r"""
## 3. 🧪 Experimento: un generador defectuoso (RANDU)

No todos los generadores son buenos. **RANDU** (IBM, años 60) se usó durante décadas en simulaciones científicas:

$$
X_{n+1} = 65\,539 \cdot X_n \bmod 2^{31}
$$

Parece inocente, pero como $65\,539 = 2^{16} + 3$, un poco de álgebra muestra que cada tres números consecutivos
cumplen:

$$
X_{n+2} \;=\; 6\,X_{n+1} \;-\; 9\,X_n \pmod{2^{31}}
$$

Es decir, los tripletes $(U_n, U_{n+1}, U_{n+2})$ **no llenan el cubo**: caen en unos pocos **planos paralelos**
cuyo vector normal es $(9, -6, 1)$. Si "miramos" la nube de puntos en esa dirección, el defecto salta a la vista.

¿Por qué importa? Imagine que simula la posición 3D de moléculas en una célula usando tres números aleatorios
seguidos como coordenadas $(x, y, z)$. Con RANDU, las moléculas sólo podrían estar en 15 "láminas" del espacio y
nunca entre ellas: cualquier cálculo de distancias, colisiones o difusión estaría sesgado, y nada en los números
individuales lo delataría. Marsaglia lo publicó en 1968 con el título *"Random numbers fall mainly in the planes"*.

> 🤔 **Antes de ejecutar, prediga:** si proyectamos los tripletes sobre la dirección $(9, -6, 1)$, ¿qué veremos
> para un buen generador? ¿Y para RANDU?
""")

nb.code(r'''
def randu(seed: int, n: int) -> np.ndarray:
    x, out = seed, np.empty(n)
    for i in range(n):
        x = (65_539 * x) % 2**31
        out[i] = x / 2**31
    return out

n_pts = 30_000
u_randu = randu(1, n_pts + 2)
u_pcg = np.random.default_rng(1).random(n_pts + 2)

def triplets(u):
    return np.column_stack([u[:-2], u[1:-1], u[2:]])

# Base ortonormal: n̂ (normal a los planos) y e (una dirección dentro de los planos)
n_hat = np.array([9, -6, 1]) / np.linalg.norm([9, -6, 1])
e = np.cross(n_hat, [0, 0, 1]); e /= np.linalg.norm(e)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, u, name in [(axes[0], u_pcg, "PCG64 (numpy)"), (axes[1], u_randu, "RANDU (IBM, 1960s)")]:
    t = triplets(u)
    ax.scatter(t @ n_hat, t @ e, s=0.6, color=ec.BLUE, alpha=0.5, lw=0, rasterized=True)
    ax.set_title(name, fontsize=12.5)
    ax.set_xlabel("proyección sobre la normal (9, −6, 1)")
    ax.grid(False)
axes[0].set_ylabel("proyección dentro del plano")
axes[1].text(0.97, 0.97, "los puntos caen en\n15 planos paralelos", transform=axes[1].transAxes,
             ha="right", va="top", fontsize=10.5, color=ec.INK)
ec.fig_title(fig, "RANDU no es aleatorio: sus tripletes viven en 15 planos",
             "30 000 tripletes (Uₙ, Uₙ₊₁, Uₙ₊₂) vistos de canto · un buen generador llena el espacio de manera uniforme")
plt.show()
''')

nb.md(r"""
> 🔎 **Moraleja.** Un generador aleatorio es un **instrumento de laboratorio**: si está descalibrado, todos los
> resultados que dependen de él están sesgados. Use siempre los generadores modernos de `numpy`.

### 🖱️ Gire usted mismo el cubo

En este gráfico 3D interactivo puede **arrastrar** para girar, usar la rueda del ratón para acercarse y hacer clic
en la leyenda para mostrar u ocultar cada generador. Con RANDU visible, gire lentamente el cubo hasta que la nube
"se abra" en láminas; luego compare con PCG64, que llena el cubo uniformemente desde cualquier ángulo.
""")

nb.code(r'''
import plotly.graph_objects as go

tr_randu, tr_pcg = triplets(u_randu)[:4000], triplets(u_pcg)[:4000]
fig = go.Figure()
fig.add_scatter3d(x=tr_randu[:, 0], y=tr_randu[:, 1], z=tr_randu[:, 2], mode="markers", name="RANDU",
                  marker=dict(size=1.8, color=ec.BLUE, opacity=0.7))
fig.add_scatter3d(x=tr_pcg[:, 0], y=tr_pcg[:, 1], z=tr_pcg[:, 2], mode="markers", name="PCG64 (numpy)",
                  marker=dict(size=1.8, color=ec.ORANGE, opacity=0.7), visible="legendonly")
fig.update_layout(title="Tripletes (Uₙ, Uₙ₊₁, Uₙ₊₂): arrastre para girar el cubo", height=620,
                  scene=dict(xaxis_title="Uₙ", yaxis_title="Uₙ₊₁", zaxis_title="Uₙ₊₂", aspectmode="cube"),
                  legend=dict(x=0.02, y=0.95), margin=dict(l=0, r=0, t=60, b=0))
fig.show()
''')

nb.md(r"""
### 🎬 Animación: girando el cubo de RANDU

Los tripletes en 3D parecen una nube uniforme… hasta que giramos el cubo y, en cierto ángulo, aparecen los planos.
Esa es la dirección de cámara perpendicular al vector $(9, -6, 1)$.
""")

nb.md(gif("modulo-00-preparacion", "0.3_randu_cubo",
          "Vista previa: al girar el cubo de tripletes de RANDU aparecen los 15 planos."))

nb.code(r'''
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

t = triplets(u_randu)[:3500]
fig = plt.figure(figsize=(6.2, 6.2))
ax3 = fig.add_subplot(projection="3d")
ax3.set_box_aspect(None, zoom=0.85)                       # deja margen para las etiquetas de los ejes
ax3.scatter(t[:, 0], t[:, 1], t[:, 2], s=3, color=ec.BLUE, alpha=0.7, lw=0)
ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])
ax3.set_xlabel("Uₙ"); ax3.set_ylabel("Uₙ₊₁"); ax3.set_zlabel("Uₙ₊₂")
for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
    axis.set_pane_color((0.97, 0.97, 0.96, 1))
ax3.set_title("RANDU en 3D: gire y aparecen los planos", loc="left", fontsize=13)

# Recorrido de cámara: de una vista "inocente" a la vista de canto de los planos
azims = np.concatenate([np.linspace(-30, 57, 30), np.full(8, 57)])
elevs = np.concatenate([np.linspace(30, 6, 30), np.full(8, 6)])

def update(f):
    ax3.view_init(elev=elevs[f], azim=azims[f])
    return ()

ec.animate(fig, update, frames=len(azims), interval=110, name="0.3_randu_cubo")
''')

nb.md(r"""
## 4. 🧪 Experimento Monte Carlo: lecturas sin errores

Un secuenciador Illumina se equivoca en aproximadamente $\varepsilon \approx 0.1\text{–}1\,\%$ de las bases. Si una
lectura (*read*) mide $L = 150$ bases, ¿qué fracción de lecturas sale **perfecta**, sin ningún error?

**Respuesta teórica.** Si cada base es correcta con probabilidad $1-\varepsilon$, independientemente de las demás:

$$
p_{\text{perfecta}} \;=\; \underbrace{(1-\varepsilon)}_{\text{base 1 correcta}} \times \underbrace{(1-\varepsilon)}_{\text{base 2 correcta}}
\times \cdots \;=\; (1-\varepsilon)^{L}
$$

Con $\varepsilon = 0.01$ y $L = 150$: $\;0.99^{150} \approx 0.221$. ¡Sólo 1 de cada 5 lecturas es perfecta!

**Ejemplo a mano para ganar intuición.** Con lecturas de sólo $L = 3$ bases: $0.99^3 = 0.99 \times 0.99 \times 0.99
\approx 0.970$. Cada base "cobra" un 1 % de la probabilidad restante; con 150 bases esas pequeñas pérdidas se
acumulan multiplicativamente. Un atajo útil: para $\varepsilon$ pequeño, $(1-\varepsilon)^L \approx e^{-\varepsilon L}$,
y $e^{-1.5} \approx 0.22$.

**Respuesta por simulación (Monte Carlo).** Simulamos $N$ lecturas y contamos cuántas no tienen errores. La
estimación $\hat p$ tiene una incertidumbre dada por el **error estándar**:

$$
\mathrm{SE}(\hat p) \;=\; \frac{\sigma}{\sqrt{N}} \;=\; \sqrt{\frac{p\,(1-p)}{N}}
$$

| Símbolo | Significado |
|---|---|
| $N$ | número de réplicas simuladas (lecturas) |
| $\hat p$ | fracción de lecturas perfectas en la simulación |
| $\sigma = \sqrt{p(1-p)}$ | desviación estándar de **una** réplica (vale 1 si es perfecta, 0 si no) |
| $\mathrm{SE}$ | cuánto variaría $\hat p$ si repitiéramos la simulación con otra semilla |

Es la misma ley que encontramos en la Lección 0.1 con el GC de una ventana: **para reducir el error a la mitad, hay
que simular cuatro veces más**. Y una distinción crucial: una semilla fija no hace "más exacto" el resultado, lo
hace **repetible**. La exactitud la da $N$; la repetibilidad, la semilla.

### 🖱️ Explore la fórmula

Antes de simular, explore la fórmula teórica. Cada curva corresponde a una tasa de error; pase el cursor para leer
la fracción de lecturas perfectas para cualquier longitud. Compare una lectura corta de Illumina (150 nt) con una
lectura larga (miles de nt).
""")

nb.code(r'''
Ls = np.arange(1, 1001)
fig = go.Figure()
for eps_i, color in zip([0.001, 0.005, 0.01, 0.05], [ec.BLUE, ec.AQUA, ec.ORANGE, ec.RED]):
    fig.add_scatter(x=Ls, y=(1 - eps_i) ** Ls, mode="lines", line=dict(color=color, width=2.5),
                    name=f"ε = {eps_i:.1%}",
                    hovertemplate=f"ε = {eps_i:.1%}<br>L = %{{x}} nt<br>P(perfecta) = %{{y:.3f}}<extra></extra>")
fig.add_vline(x=150, line_dash="dot", line_color=ec.MUTED, annotation_text="Illumina 150 nt",
              annotation_position="top right")
fig.update_layout(title="La fracción de lecturas perfectas cae exponencialmente con la longitud",
                  xaxis_title="Longitud de la lectura L (nt)", yaxis_title="P(lectura sin errores) = (1 − ε)ᴸ",
                  height=460, yaxis_range=[0, 1.02], hovermode="x unified")
fig.show()
''')

nb.md(r"""
> 🤔 **Antes de ejecutar la simulación, prediga:** con sólo $N = 100$ lecturas simuladas, ¿entre qué valores podría
> caer $\hat p$? (Pista: $\mathrm{SE} = \sqrt{0.221 \times 0.779 / 100} \approx 0.04$.)

Corramos 25 simulaciones independientes (25 semillas) y veamos cómo converge cada una:
""")

nb.code(r'''
eps, L = 0.01, 150
p_theory = (1 - eps) ** L
N_max = 20_000
checkpoints = np.unique(np.logspace(1, np.log10(N_max), 60).astype(int))

fig, ax = plt.subplots(figsize=(10, 5))
for seed in range(25):
    rng = np.random.default_rng(seed)
    errors = rng.random((N_max, L)) < eps                  # True donde el secuenciador se equivoca
    perfect = ~errors.any(axis=1)                          # lectura sin ningún error
    running = np.cumsum(perfect)[checkpoints - 1] / checkpoints
    ax.plot(checkpoints, running, color=ec.BLUE, alpha=0.25, lw=1)

se = np.sqrt(p_theory * (1 - p_theory) / checkpoints)
ax.fill_between(checkpoints, p_theory - 1.96 * se, p_theory + 1.96 * se, color=ec.ORANGE, alpha=0.12, lw=0)
ax.plot(checkpoints, p_theory + 1.96 * se, color=ec.ORANGE, lw=1.4)
ax.plot(checkpoints, p_theory - 1.96 * se, color=ec.ORANGE, lw=1.4)
ax.plot([10, N_max], [p_theory, p_theory], color=ec.INK, lw=1.2)
ec.label_end(ax, N_max, p_theory, f"teoría: 0.99¹⁵⁰ = {p_theory:.3f}", color=ec.INK)
ax.annotate("banda teórica ± 1.96·SE (95 %)", xy=(40, p_theory + 1.96 * se[checkpoints.searchsorted(40)]),
            xytext=(25, 18), textcoords="offset points", fontsize=10, color=ec.INK_2,
            arrowprops=dict(arrowstyle="-", color=ec.MUTED, lw=1))
ax.set_xscale("log")
ax.set_xlim(10, N_max * 6)
ax.set_ylim(0, 0.5)
ax.set_xlabel("Número de lecturas simuladas N (escala log)")
ax.set_ylabel("Fracción de lecturas perfectas")
ec.title(ax, "Cada semilla da un camino distinto, pero todos convergen como 1/√N",
         "25 simulaciones Monte Carlo (una línea azul por semilla) · error por base ε = 1 %, lectura de 150 nt")
plt.show()
''')

nb.code(r'''
# Reproducibilidad en acción: la misma semilla da EXACTAMENTE el mismo número
def simulate_perfect_fraction(seed, n=5_000, eps=0.01, L=150):
    rng = np.random.default_rng(seed)
    return (~(rng.random((n, L)) < eps).any(axis=1)).mean()

print("semilla 1 :", simulate_perfect_fraction(1))
print("semilla 1 :", simulate_perfect_fraction(1), "← idéntico")
print("semilla 2 :", simulate_perfect_fraction(2), "← distinto, pero dentro del error estándar")
print(f"SE teórico con N = 5000: {np.sqrt(p_theory * (1 - p_theory) / 5000):.4f}")
''')

nb.md(r"""
> ✅ **Compruebe su comprensión.** (1) Si dos compañeros corren la misma simulación con semillas distintas y
> obtienen 0.212 y 0.224, ¿alguno se equivocó? (2) ¿Cuántas lecturas más necesitaría simular para que la diferencia
> típica entre semillas fuera 10 veces menor?
""")

nb.md(r"""
## 5. Registrar el entorno: versiones de paquetes

En la sección de métodos de un artículo usted reporta "centrifugado a 4 °C, 5 000 × g, rotor JA-25.50". Nadie
escribiría sólo "centrifugado un rato". En bioinformática el equivalente es: **Python 3.12, numpy 2.0, pandas 2.2,
samtools 1.19…** Las versiones importan porque el comportamiento por defecto de las funciones cambia entre versiones
(un ejemplo real: en `pandas` 2.0 cambió el valor por defecto de `numeric_only` en agregaciones como
`df.groupby(...).mean()`, y códigos que antes ignoraban en silencio las columnas de texto empezaron a fallar).

Colab actualiza sus paquetes varias veces al año: el mismo notebook puede comportarse distinto dentro de seis meses.
Registre siempre el entorno al inicio o al final del análisis:
""")

nb.code(r'''
env = ec.environment_report()
pd.Series(env, name="versión").to_frame()
''')

nb.code(r'''
# Guardar la lista completa de paquetes (para reinstalar exactamente lo mismo: pip install -r requirements.txt)
from importlib.metadata import distributions
freeze = "\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions()))
open("requirements_snapshot.txt", "w").write(freeze + "\n")
print(f"{len(freeze.splitlines())} paquetes registrados en requirements_snapshot.txt. Primeras líneas:")
print("\n".join(freeze.splitlines()[:8]))
''')

nb.md(r"""
Para herramientas de línea de comandos se usan **entornos conda** (`environment.yml`) o **contenedores**
(Docker/Apptainer), que congelan el sistema completo. Los veremos en el Módulo 18.

## 6. 🧪 Huellas digitales de archivos: *hashes*

Usted descarga un genoma de 3 GB. ¿Cómo sabe que llegó **completo y sin cambios**? ¿Y que su colega usa
**exactamente** el mismo archivo?

Una **función hash** produce una especie de **huella dactilar** del archivo: un número corto (64 caracteres
hexadecimales en SHA-256) que se calcula a partir de **todo** el contenido. Igual que con las huellas dactilares,
dos archivos idénticos tienen la misma huella, y es prácticamente imposible que dos archivos distintos compartan
una. Lo sorprendente es que si cambia **un solo bit** del archivo, la huella cambia por completo, no "un poquito".

| Propiedad | Qué significa en la práctica |
|---|---|
| Determinista | el mismo archivo da siempre la misma huella, en cualquier computadora |
| Tamaño fijo | un archivo de 1 KB o de 100 GB da una huella de 256 bits |
| Efecto avalancha | un cambio mínimo cambia ~la mitad de los bits de la huella |
| Irreversible | a partir de la huella no se puede reconstruir el archivo |

> 🤔 **Antes de ejecutar, prediga:** si cambiamos 1 base de 100 000 en un genoma, ¿cuántos de los 256 bits de la
> huella cambiarán? ¿1? ¿10? ¿128?

Las bases de datos (NCBI, Ensembl) publican los **MD5/SHA-256** de sus archivos precisamente para esto.
""")

nb.code(r'''
import hashlib

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

rng = np.random.default_rng(0)
genome_seq = "".join(rng.choice(list("ACGT"), 100_000))
mutated = genome_seq[:50_000] + ("A" if genome_seq[50_000] != "A" else "C") + genome_seq[50_001:]

h1, h2 = sha256(genome_seq.encode()), sha256(mutated.encode())
print("original :", h1)
print("1 SNP    :", h2)
print("¿iguales?:", h1 == h2)
''')

nb.md(r"""
Visualicemos el **efecto avalancha**: comparamos los 256 bits de ambas huellas. Un buen hash cambia
**aproximadamente la mitad** de los bits ante un cambio mínimo, de modo que es imposible "adivinar" el archivo a
partir de la huella o fabricar otro archivo con la misma huella.
""")

nb.code(r'''
def to_bits(hex_digest: str) -> np.ndarray:
    return np.array([int(b) for b in bin(int(hex_digest, 16))[2:].zfill(256)]).reshape(8, 32)

b1, b2 = to_bits(h1), to_bits(h2)
diff = b1 != b2

from matplotlib.colors import ListedColormap
fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.2))
panels = [(b1, "Huella del genoma original", ListedColormap([ec.GRID, ec.BLUE])),
          (b2, "Huella tras cambiar 1 base de 100 000", ListedColormap([ec.GRID, ec.BLUE])),
          (diff, f"Bits distintos: {diff.sum()} de 256 ({diff.mean():.0%})", ListedColormap([ec.GRID, ec.ORANGE]))]
for ax, (m, title, cmap) in zip(axes, panels):
    ax.imshow(m, cmap=cmap, aspect="equal", vmin=0, vmax=1)
    ax.set_xticks(np.arange(-0.5, 32, 1), minor=True); ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax.grid(which="minor", color=ec.SURFACE, linewidth=1.5); ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(title, fontsize=11.5, loc="left")
ec.fig_title(fig, "Efecto avalancha: un SNP cambia ~la mitad de los bits de SHA-256",
             "Cada cuadro es un bit (azul = 1) · la huella completa tiene 256 bits")
plt.show()
''')

nb.md(r"""
> 🔎 **Qué observamos.** Cambió cerca de la mitad de los bits, repartidos sin patrón. No hay forma de saber, mirando
> la huella, *qué* cambió ni *dónde*: sólo *que algo* cambió. Eso es exactamente lo que necesitamos para verificar
> integridad.
>
> ✅ **Compruebe su comprensión.** Un colega le envía un FASTQ de 20 GB y el MD5 publicado por el secuenciador. Usted
> calcula el MD5 del archivo recibido y difiere en un solo carácter del publicado. ¿El archivo está "casi bien"?
""")

nb.code(r'''
# En la práctica: verificar archivos en disco, leyendo por bloques (sirve para archivos de GB)
def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

open("genome_demo.fasta", "w").write(">demo\n" + genome_seq + "\n")
print("SHA-256 de genome_demo.fasta:", file_sha256("genome_demo.fasta"))
# En la terminal de Linux/Colab el equivalente es:
!sha256sum genome_demo.fasta 2>/dev/null || shasum -a 256 genome_demo.fasta
''')

nb.md(r"""
## 7. Git y GitHub: la máquina del tiempo del análisis

Seguramente ha visto carpetas con archivos como `analisis_final.py`, `analisis_final_v2.py`,
`analisis_final_v2_AHORA_SI.py`… Git resuelve ese caos. Funciona como un **cuaderno de laboratorio con fotografías
fechadas**: cada vez que usted hace un **commit**, Git toma una foto de todos sus archivos y la guarda con un mensaje
("agrego filtrado de calidad"), su autor y su fecha. Puede volver a cualquier foto anterior, ver exactamente qué
líneas cambiaron entre dos fotos, o trabajar en una **rama** experimental sin arruinar la versión que funciona.
**GitHub** es la biblioteca en la nube donde se guarda y comparte ese cuaderno (este curso vive en GitHub).

| Concepto | Qué es |
|---|---|
| **Repositorio** | la carpeta del proyecto + todo su historial |
| **Commit** | una "foto" del proyecto con un mensaje y un identificador único (¡un *hash*!) |
| **Rama** (*branch*) | una línea de trabajo paralela (p. ej. probar otro alineador) |
| **Merge** | unir una rama de vuelta a la principal |
| **Push / Pull** | subir / bajar commits de GitHub |
""")

nb.code(r'''
# Diagrama: historia de un análisis con una rama experimental
fig, ax = plt.subplots(figsize=(11, 3.4))
main_x = [0, 1, 2, 5, 6]
feat_x = [3, 4]
ax.plot([0, 2], [0, 0], color=ec.BLUE, lw=3, zorder=1)
ax.plot([2, 5, 6], [0, 0, 0], color=ec.BLUE, lw=3, zorder=1)
ax.plot([2, 3, 4, 5], [0, 1, 1, 0], color=ec.ORANGE, lw=3, zorder=1)
msgs_main = ["datos crudos", "control de\ncalidad", "alineamiento\n(BWA)", "merge", "figuras\nfinales"]
msgs_feat = ["probar\nminimap2", "comparar\nresultados"]
for x, m in zip(main_x, msgs_main):
    ax.scatter(x, 0, s=260, color=ec.BLUE, edgecolor=ec.SURFACE, linewidth=3, zorder=3)
    ax.text(x, -0.32, m, ha="center", va="top", fontsize=9.5, color=ec.INK_2)
for x, m in zip(feat_x, msgs_feat):
    ax.scatter(x, 1, s=260, color=ec.ORANGE, edgecolor=ec.SURFACE, linewidth=3, zorder=3)
    ax.text(x, 1.3, m, ha="center", va="bottom", fontsize=9.5, color=ec.INK_2)
ax.text(6.35, 0, "main", va="center", fontsize=11, fontweight="bold", color=ec.INK)
ax.text(4.35, 1, "experimento", va="center", fontsize=11, fontweight="bold", color=ec.INK)
ax.set_xlim(-0.4, 7.3); ax.set_ylim(-1.1, 2.0); ax.axis("off")
ax.set_title("Git guarda cada paso del análisis; las ramas permiten experimentar sin riesgo", loc="left")
plt.show()
''')

nb.md(r"""
Hagamos un mini-repositorio aquí mismo (Git viene instalado en Colab):
""")

nb.code(r'''
import shutil, tempfile
start_dir = os.getcwd()
repo = tempfile.mkdtemp(prefix="demo_repo_")
os.chdir(repo)
GIT = "git -c user.name=Estudiante -c user.email=estudiante@ejemplo.com"

open("analysis.py", "w").write("gc = 0.50\n")
!git init -q -b main && {GIT} add analysis.py && {GIT} commit -q -m "Primer análisis: GC fijo"
open("analysis.py", "w").write("gc = 0.508  # valor real de E. coli\n")
!{GIT} commit -q -am "Uso el GC real de E. coli"
!git log --oneline
print("\n--- ¿Qué cambió entre las dos fotos? ---")
!git --no-pager diff HEAD~1 HEAD
os.chdir(start_dir)
''')

nb.md(r"""
> 🔎 Observe que cada commit tiene un identificador como `a3f9c12`: es el comienzo de un **hash SHA-1** de su
> contenido. ¡La sección 6 y la 7 son la misma idea!

### Colab ↔ GitHub

* **Abrir** un notebook de GitHub en Colab: `https://colab.research.google.com/github/<usuario>/<repo>/blob/main/<archivo>.ipynb`
  (así funcionan los botones *Open in Colab* de este curso).
* **Guardar** su versión: `Archivo → Guardar una copia en GitHub` (autorice su cuenta la primera vez).
* **Datos persistentes**: monte su Google Drive para que los archivos sobrevivan al reinicio de la sesión:
""")

nb.code(r'''
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")      # aparecerá una ventana para autorizar el acceso
else:
    print("(Sólo en Colab) Aquí se montaría Google Drive en /content/drive")
''')

nb.md(r"""
## ✅ Lista de verificación de reproducibilidad

Basada en Sandve *et al.* (2013), *Ten Simple Rules for Reproducible Computational Research*:

- [ ] Toda aleatoriedad usa un `rng = np.random.default_rng(SEMILLA)` explícito.
- [ ] Registro las versiones del software (`environment_report()`, `pip freeze`, `tool --version`).
- [ ] Anoto el origen exacto de los datos (número de acceso, versión, fecha) y verifico su *hash*.
- [ ] Nunca edito los datos crudos a mano: todo cambio es código.
- [ ] Cada figura del informe se genera con código versionado en Git.
- [ ] El notebook corre de principio a fin con `Reiniciar y ejecutar todo`.

## ✍️ Ejercicios

**Ejercicio 1 — Período de un LCG.** Con $a = 5$, $c = 1$, $m = 16$ y semilla 0, ¿cada cuántos pasos se repite la
sucesión? ¿Y con $a = 4$? (La teoría de Hull–Dobell dice cuándo el período es máximo, $m$.)

**Ejercicio 2 — Diseño de una simulación.** ¿Cuántas lecturas $N$ necesita simular para estimar $p_\text{perfecta}$
con un error estándar menor a 0.001? Despeje $N$ de la fórmula del SE.

**Ejercicio 3 — Lecturas largas.** Repita el cálculo teórico para lecturas de Nanopore de 10 000 nt con
$\varepsilon = 5\,\%$. ¿Qué implica para el análisis de estas lecturas?

**Ejercicio 4 — Hash de datos reales.** Calcule el SHA-256 del archivo GenBank de SARS-CoV-2 que descargó en la
Lección 0.2 y compárelo con el de un compañero. ¿Coinciden? ¿Por qué podrían no coincidir aunque el genoma sea el
mismo? (Pista: mire la fecha en la primera línea del archivo.)
""")

nb.code(r'''
#@title 🔑 Solución — Ejercicio 1 { display-mode: "form" }
def lcg_period(a, c, m, seed=0):
    seen, x, n = {}, seed, 0
    while x not in seen:
        seen[x] = n; x = (a * x + c) % m; n += 1
    return n - seen[x]
print("a=5: período =", lcg_period(5, 1, 16), "(máximo = 16)")
print("a=4: período =", lcg_period(4, 1, 16))
''')

nb.code(r'''
#@title 🔑 Solución — Ejercicio 2 { display-mode: "form" }
# SE = sqrt(p(1-p)/N) < 0.001  =>  N > p(1-p)/0.001^2
N_needed = int(np.ceil(p_theory * (1 - p_theory) / 0.001**2))
print(f"N > {N_needed:,} lecturas")
''')

nb.code(r'''
#@title 🔑 Solución — Ejercicio 3 { display-mode: "form" }
p_np = (1 - 0.05) ** 10_000
print(f"p(perfecta) = {p_np:.2e}  → prácticamente ninguna lectura larga es perfecta;")
print("los análisis con lecturas largas deben tolerar errores (alineadores y ensambladores especiales, pulido).")
''')

nb.md(r"""
## 📌 Resumen

* La computadora produce números **pseudoaleatorios**; la **semilla** fija el punto de partida y hace repetible
  cualquier simulación. Use `np.random.default_rng(semilla)`.
* Un mal generador (RANDU) sesga los resultados: los generadores son instrumentos que hay que "calibrar".
* El error de una simulación Monte Carlo cae como $1/\sqrt{N}$; la semilla da **repetibilidad**, no exactitud.
* Registre **versiones** del software y **hashes** de los datos.
* **Git** guarda la historia del análisis; GitHub la comparte; Colab abre notebooks directamente desde GitHub.

**Próximo módulo (1):** Biología molecular para bioinformáticos — el dogma central como sistema de información.

## 📚 Para profundizar

* Sandve, G. K. *et al.* (2013). Ten Simple Rules for Reproducible Computational Research. *PLoS Comput Biol* 9(10): e1003285.
* Marsaglia, G. (1968). Random numbers fall mainly in the planes. *PNAS* 61(1): 25–28.
* O'Neill, M. E. (2014). *PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation.*
* Perez-Riverol, Y. *et al.* (2016). Ten Simple Rules for Taking Advantage of Git and GitHub. *PLoS Comput Biol* 12(7): e1004947.
""")

import os; nb.save(os.path.join(os.environ.get("NB_ROOT", "/Users/juvenalyosa/bioinformatics"), PATH))
print("saved", PATH)
