# FDASRSF — Diagnose & Empfehlung

**Stand:** 2026-05-04
**Zweck:** Reine Recherche-Doku. **Keine** Installations-Aktion in dieser Session ausgeführt.

---

## 1. Versions-Status (read-only festgestellt)

| Paket | Version | Pfad |
|---|---|---|
| Python | 3.10 | `/usr/bin/python3` |
| numpy | 1.26.4 | (in `~/.local/...`) |
| scipy | **1.15.1** | `~/.local/lib/python3.10/site-packages/` |
| matplotlib | **3.10.9** | `~/.local/lib/python3.10/site-packages/` |
| **mpl_toolkits** (apt) | **3.5.1** | `/usr/lib/python3/dist-packages/` (apt: `python3-matplotlib 3.5.1-2build1`) |
| fdasrsf | **2.6.9** (installiert) | `~/.local/lib/python3.10/site-packages/` |

---

## 2. KORREKTUR DER URSPRÜNGLICHEN ANNAHME

> **Behauptung:** „fdasrsf 2.6.9 hat einen `from matplotlib import docstring`-Bug."
> **Realität:** Falsch zugeordnet. Der Bug liegt **nicht** in fdasrsf.

### Wahre Ursachenkette (verifiziert per Traceback)

```
fdasrsf/__init__.py
  → fdasrsf/time_warping.py (Z. 14)
    → fdasrsf/PPD.py (Z. 7)
      → from mpl_toolkits.mplot3d import Axes3D
        → /usr/lib/python3/dist-packages/mpl_toolkits/mplot3d/axes3d.py (Z. 23)
          → from matplotlib import _api, cbook, docstring, _preprocess_data
            → ImportError: cannot import name 'docstring' from 'matplotlib'
```

### Diagnose

`mpl_toolkits` ist im **System-apt-Paket `python3-matplotlib 3.5.1`** unter `/usr/lib/python3/dist-packages/mpl_toolkits/` installiert und enthält ein **regulares `__init__.py`** (kein PEP 420 namespace package). Damit „gewinnt" das alte System-Verzeichnis bei `import mpl_toolkits` und überschattet die User-Site-Variante in `~/.local/lib/python3.10/site-packages/mpl_toolkits/` (3.10.9).

Verifiziert:
```python
>>> import mpl_toolkits
>>> mpl_toolkits.__path__
['/usr/lib/python3/dist-packages/mpl_toolkits']   # nur das System-Verzeichnis!
```

Das alte `axes3d.py` (matplotlib 3.5.1) referenziert `matplotlib.docstring` — das ist in matplotlib 3.10 nach `matplotlib._docstring` umbenannt → **ImportError**.

### Konsequenz

* fdasrsf 2.6.9 ist **bezüglich `docstring` sauber** (`grep` findet keine `docstring`-Imports im Paket).
* fdasrsf 2.6.9 nutzt korrekt `scipy.integrate.trapezoid` / `cumulative_trapezoid` → **kompatibel mit scipy 1.15**.
* Der einzige fdasrsf-eigene Bezug auf `mpl_toolkits` sind **zwei Plot-Module**:
  - `fdasrsf/PPD.py:7` (Persistent-Peak-Diagramme)
  - `fdasrsf/boxplots.py:12` (Funktional-Boxplots)
* `PPD.py` wird leider von `time_warping.py` per **Top-Level-Import** gezogen → blockiert das ganze Paket.

---

## 3. fdasrsf Versions-Range × Kompatibilität

| fdasrsf | scipy ≤ 1.13 (`trapz` da) | scipy ≥ 1.14 (`trapz` weg) | Bemerkung |
|---|---|---|---|
| **2.4.x** (Feb 2023 – Aug 2023) | OK | **BROKEN** (ImportError `trapz`) | „Klassische" Phase |
| **2.5.x** (Nov 2023 – Jul 2024) | OK | **BROKEN** | Letzte 2.5: 2.5.14 (4. Jul 2024) |
| **2.6.0+** (5. Jul 2024 ff.) | (vermutlich OK¹) | **OK** | Issue #57 als Trigger; auch numpy 2.0 (Issue #56) |
| **2.6.9** (28. Mar 2026, aktuell) | OK | **OK** | KEIN docstring-Bug intern |

¹ 2.6.x wurde primär für scipy ≥ 1.14 gefixt; Rückwärtskompatibilität zu scipy ≤ 1.13 war Bug-Stand Issue #57 nicht das Ziel. In der Praxis irrelevant.

**Quellen:**
* PyPI Release-History (https://pypi.org/project/fdasrsf/#history)
* Issue #57 „Support SciPy 1.14" — geschlossen 2024-07-05 mit Kommentar „fixed in master and new version released" → korrespondiert exakt mit Release 2.6.0 am 2024-07-05
* Issue #56 „Support for NumPy 2.0" — ebenfalls 2024-07-05 fixed

**Es gibt KEIN offenes/geschlossenes fdasrsf-Issue zu „matplotlib docstring"** — weil der Bug nicht in fdasrsf liegt.

---

## 4. Multivariate / Joint-Warp API in fdasrsf

Für „gemeinsames φ(t) für 2 PCs" ist die Klasse `fdacurve` (in `curve_stats.py`) zuständig.

```python
from fdasrsf import fdacurve
# beta: ndarray shape (n_curves, M_dim, N_samples)
# Für 2 PCs einer einzigen Trajektorie:
#   beta = data[None, :, :]   mit data.shape == (2, T)
#   → (1, 2, T)  — eine Kurve in 2D
fc = fdacurve(beta, mode='O', N=200, scale=False)
fc.srvf_align(rotation=False, lam=0.0, parallel=False, method='DP')
# → fc.gams, fc.betan, fc.qn etc.
```

`srvf_align` und `multiple_align_curves` produzieren **EIN** gemeinsames γ pro Kurve über alle Dimensionen — genau die gewünschte Semantik.

**Aber:** Beide Methoden hängen indirekt am `time_warping.py`/`PPD.py`-Pfad → blockiert solange der `mpl_toolkits`-Konflikt besteht.

---

## 5. Empfehlungen — Reihenfolge der Praktikabilität

### EMPFEHLUNG 1 (BEST): apt-Paket `python3-matplotlib` deinstallieren

Aufwand: 1 Befehl. Risiko: gering, aber **DARF NUR DER USER ENTSCHEIDEN**.

```bash
# NICHT VON CLAUDE AUSFÜHREN!
sudo apt remove python3-matplotlib python-matplotlib-data
# Damit verschwindet /usr/lib/python3/dist-packages/mpl_toolkits/
# → user-site mpl_toolkits 3.10.9 kommt zum Zug
# → fdasrsf 2.6.9 importiert sauber
```

**Vorteil:** Behebt eine Klasse von Folgekonflikten dauerhaft. matplotlib + mpl_toolkits kommen dann konsistent aus pip user-site.
**Risiko:** Falls andere apt-Pakete (system-tools, zB `pulseview`, `cura`, `ipython3` aus apt) das python3-matplotlib hart als Dependency verlangen, würden sie gleich mit-deinstalliert. **`apt remove --simulate` zuerst!**

### EMPFEHLUNG 2: Lokaler Wrapper, der `sys.path` vor dem Import korrigiert

Aufwand: ~15 Zeilen. Risiko: kosmetisch.

Da `mpl_toolkits` ein reguläres Package mit `__init__.py` im System ist, hilft simple `sys.path`-Reorder NICHT (siehe Recherche). Aber: man kann `mpl_toolkits` direkt aus `sys.modules` entfernen und manuell aus user-site laden, BEVOR fdasrsf importiert wird:

```python
# Skizze (NICHT zum Ausführen jetzt, nur Vorschlag):
import sys, importlib, importlib.util, os
USER_MPL = "/home/hz/.local/lib/python3.10/site-packages/mpl_toolkits"
spec = importlib.util.spec_from_file_location(
    "mpl_toolkits", os.path.join(USER_MPL, "__init__.py"),
    submodule_search_locations=[USER_MPL])
mod = importlib.util.module_from_spec(spec)
sys.modules["mpl_toolkits"] = mod
spec.loader.exec_module(mod)
# DANACH:
import fdasrsf  # zieht jetzt das richtige mpl_toolkits
```

**Vorteil:** Kein System-Eingriff.
**Nachteil:** Hack, fragil, muss in jedes CLI rein.

### EMPFEHLUNG 3: Eigener venv für genau dieses CLI

Aufwand: ~5 Min. Risiko: null (System unverändert).

```bash
# VORSCHLAG, NICHT AUSGEFÜHRT:
python3 -m venv /home/hz/Data/Attractor/.venv-fdasrsf --system-site-packages=False
source /home/hz/Data/Attractor/.venv-fdasrsf/bin/activate
pip install numpy==1.26.4 scipy==1.15.1 matplotlib==3.10.9 fdasrsf==2.6.9
# Im CLI: shebang #!/home/hz/Data/Attractor/.venv-fdasrsf/bin/python
```

**Vorteil:** Vollständig isoliert, keine Berührung mit System-`mpl_toolkits`.
**Nachteil:** ssmlearnpy/tensorflow/pymc müssten im venv ggf. mit-installiert werden, falls das CLI sie auch braucht.

### EMPFEHLUNG 4 (Fallback): Manuelle SRVF-Implementierung

Aufwand: ~200–350 Zeilen numpy. Risiko: Mathematische Korrektheit muss validiert werden.

Nur sinnvoll, wenn EMPFEHLUNGEN 1–3 alle nicht gewünscht sind.

---

## 6. Manuelle SRVF-Implementierung — Skizze

**Wenn nichts anderes geht.** Pseudocode für eine minimal-funktionale, multivariate joint-Warp-Routine:

### Funktions-Signaturen (Skizze, NICHT implementiert)

```python
def srvf(f, t):
    """SRVF q(t) = f'(t) / sqrt(|f'(t)|), multivariat.
    f: (D, T)  → q: (D, T), eps für f'≈0."""

def inv_srvf(q, t, f0):
    """Rekonstruktion: f(t) = f0 + ∫ q(s)|q(s)| ds."""

def warp_apply(f, gamma, t):
    """f ∘ γ via Interpolation. Multivariat: jede Dim einzeln."""

def srvf_distance(q1, q2, t):
    """L2-Norm der SRVFs (rotationsinvariant nur bei Curve-Mode)."""

def joint_dp_warp(q1, q2, t, lam=0.0, grid=7):
    """Dynamic Programming auf SRVF-Kostenfunktion.
    q1, q2: (D, T). Kosten = ||q1(t) - sqrt(γ') q2(γ(t))||^2 (über alle Dim summiert).
    Return: γ als (T,)-Vektor.
    Standard-Algorithmus: Bellman, Restriktion auf monotone γ' ∈ [0, grid].
    Referenz: Srivastava & Klassen Buch, Algorithmus 8.1 / Listing in
              https://github.com/jdtuck/fdasrsf_python/blob/master/src/DynamicProgrammingQ2.c"""

def pair_align_multivariate(f1, f2, t, lam=0.0, grid=7):
    """Wrapper: SRVFs berechnen, joint_dp_warp, Resultat zurückwarpen."""
```

### Kernalgorithmen

1. **SRVF + Inverse:** Trivial, je ~5 Zeilen numpy.
2. **DP-Warp:** Der einzige nicht-triviale Teil. Original-C-Code in `src/DynamicProgrammingQ2.c` (~300 Zeilen C). Eine reine Python/numpy-Variante mit Numba-JIT ist ca. **150–200 Zeilen**, ohne Numba ~3-5× langsamer aber für T~1000 noch erträglich.
3. **Multivariater Cost:** `cost(s,t,s',t') = sum_dim ∫ (q1[d, s..t] − √γ' · q2[d, s'..t'])^2 ds`. Genau wie univariat, nur Summe über Dim.

### Code-Umfang Schätzung

| Komponente | Zeilen (numpy) |
|---|---|
| `srvf` / `inv_srvf` / `warp_apply` | ~30 |
| DP-Warp (univariat → multivariat triviale Erweiterung) | ~150 |
| Pairwise + Tests | ~50 |
| **Gesamt** | **~230 Zeilen** |

### Referenz-Implementierungen

* **Original C-Code:** https://github.com/jdtuck/fdasrsf_python/blob/master/src/DynamicProgrammingQ2.c — direkt portierbar
* **MATLAB-Original:** http://ssamg.stat.fsu.edu/software (Srivastava/Klassen FSU-Toolbox)
* **Lehr-Implementation:** Tucker et al., „Generative Models for Functional Data Using Phase and Amplitude Separation", Comp. Stat. Data Anal. 61 (2013), Anhang
* **Python-Snippet:** https://github.com/glemaitre/srsf-python (kleines Demo, 1D)

---

## 7. Alternativen-Bewertung (<150 Wörter)

* **dtw-python:** Univariate DTW mit `mvm`-Step-Pattern liefert keinen gemeinsamen Warp über mehrere Kanäle. `dtw(query, reference)` mit gestackten 2D-Daten geht nur via L2-Norm-Distanz pro Zeitpunkt — das ist näher an Multidimensional-DTW, aber ohne SRVF-Phase/Amplitude-Trennung. Für H₀-Falsifikation („sind die zwei PCs nur Reparametrisierung voneinander?") **nur bedingt geeignet** — DTW kennt keine Quotient-Norm bzgl. Reparametrisierungs-Gruppe.
* **tslearn:** `dtw_path_independent` warpt jede Dimension separat → falsche Semantik. `subsequence_path` ist für Such-Probleme gedacht, nicht für Alignment.
* **Manuelle SRVF (~200–300 Zeilen):** Realistisch und mathematisch sauber. Vorteil: deterministische Kontrolle, keine Abhängigkeits-Hölle.
* **venv mit fdasrsf 2.6.9 + matplotlib 3.10:** Praktischer als 2.4.3 mit altem scipy. **Klar zu bevorzugen** vor manueller Implementierung, weil getestet & validiert.

---

## 8. KONKRET — Was Claude als Nächstes tun soll

> **Empfehlung an den User (Jesus Friedrich), in der Reihenfolge anbieten:**
>
> 1. **`apt remove --simulate python3-matplotlib`** ausführen lassen, Output prüfen, **dann entscheiden** ob die abhängigen Pakete entbehrlich sind. Falls ja → `sudo apt remove` (USER macht das, Claude NICHT). fdasrsf 2.6.9 sollte danach sofort funktionieren.
>
> 2. Falls 1) zu riskant: **venv `~/Data/Attractor/.venv-fdasrsf`** anlegen mit `python3 -m venv --without-system-site-packages`, dort `pip install fdasrsf==2.6.9 numpy==1.26.4 scipy==1.15.1 matplotlib==3.10.9` und das CLI mit dem venv-Python aufrufen. Touched nichts am System.
>
> 3. Falls 2) zu umständlich (z.B. weil das CLI auch ssmlearnpy braucht und das im venv neu installiert werden müsste): **`sys.modules`-Hack in EINEM einzigen kleinen Loader-Modul** (`_fdasrsf_loader.py`), das vor `import fdasrsf` das `mpl_toolkits` aus user-site zwangslädt. ~15 Zeilen, dann läuft fdasrsf 2.6.9.
>
> 4. **Manuelle Implementierung NUR** wenn der User das explizit will (Pädagogisches Argument oder „kein Fremd-Code"-Wunsch).

**Claude soll NICHTS davon ohne explizite User-Anweisung tun. Diese Datei ist Diagnose, nicht Aktion.**
