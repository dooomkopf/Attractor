# Review: SSM/res vs. analyze_wang/analyze_residuals Workflow

## 1. Fachlich falsche oder irreführende Teile in SSM/res/01..05

### Kritische Fehler:
- **SSM/res/03_cli_phase_lock.py:48** - Default `ssm_dim=2` ist inkonsistent und fachlich falsch für BTC-Residuen (sollte 4 sein)
- **SSM/res/05_cli_scan.py:24** - Kaputte Imports: `from analyze_residuals.ssm_learn` existiert nicht (sollte aus ssmlearn_res oder lokalen Modulen kommen)
- **SSM/res/01_cli_precheck.py:24** - Importiert `ssmlearn_res` direkt ohne SSM_res_* Module zu nutzen (inkonsistent)

### Irreführende Aspekte:
- **SSM/res/02_cli_harmonics.py:2** - Kommentar "06: Harmonics" statt "02" (Copy-Paste-Fehler)
- **SSM/res/03_cli_phase_lock.py:2** - Kommentar "08: Phase lock" statt "03"
- **SSM/res/** - Nutzt teilweise alte Konventionen (z.B. direkte `ssmlearn_res` Imports) statt modularisierte Struktur

## 2. Soll SSM/res direkt auf analyze_residuals aufsetzen?

**Empfehlung: INSPIRIERT, nicht DIREKT AUFSETZEN**

Gründe:
- `analyze_residuals` ist bereits weiterentwickelt (hat precheck.py, data.py, common.py Module)
- SSM/res hat eigene Module (SSM_res_data.py, SSM_res_embedding.py) die BTC-spezifische Features haben
- Direktes Aufsetzen würde zu doppelten Abhängigkeiten führen

Besserer Ansatz:
- SSM/res sollte seine eigenen Module behalten (SSM_res_data, SSM_res_embedding)
- Von analyze_residuals nur die **Struktur und Print-Formate** übernehmen
- Eigene Implementierung für BTC-spezifische Aspekte

## 3. Minimale zwingenden Fixes für SSM/res/01..05

```python
# 1. SSM/res/03_cli_phase_lock.py Zeile 48:
ap.add_argument('--ssm_dim', type=int, default=4)  # statt default=2

# 2. SSM/res/05_cli_scan.py Zeile 24-26:
# ENTFERNEN:
from analyze_residuals.ssm_learn import run_slave_test
from analyze_residuals.data import build_residual_context
# ERSETZEN durch:
from SSM_res_data import load_data
from SSM_res_embedding import build_embedding, pca
from ssmlearn_res import fit_ssm
# Dann run_slave_test lokal implementieren

# 3. SSM/res/02_cli_harmonics.py Zeile 2:
"""02: Harmonics — PSD per PC: which carry omega, 2*omega?"""

# 4. SSM/res/03_cli_phase_lock.py Zeile 2:
"""03: Phase lock — psi = 2*phi_main - phi_sub, resultant R, drift analysis."""
```

## 4. Gefahr der zu weit getriebenen Wang-Analogie

### Kritische Punkte wo BTC != Wang:

1. **IC (Initial Conditions)**:
   - Wang hat IC als 3D-Vektor (x,y,z)
   - BTC hat keine ICs, sondern historische Daten mit start_idx
   - GEFAHR: IC-Parameter in BTC-Scripts wäre sinnlos

2. **Transient Removal**:
   - Wang verwirft ersten Teil der Simulation (transient_frac)
   - BTC hat echte historische Daten ohne Transiente
   - GEFAHR: Frühe BTC-Daten wegwerfen wäre Datenverlust

3. **Parameter-Scans**:
   - Wang scannt physikalische Parameter (a,b,c,d)
   - BTC scannt Analyse-Parameter (ssm_dim, poly_degree)
   - GEFAHR: Versuch physikalische Konstanten in BTC zu finden

4. **Harmonische Identifikation**:
   - Wang: exakte 2:1 Resonanz durch Systemdesign
   - BTC: empirische ~2:1 Resonanz, nicht exakt
   - GEFAHR: Forcieren exakter Verhältnisse wo keine sind

5. **Amplitude Scaling**:
   - Wang: theoretisches |z_sub| ~ |z_main|²
   - BTC: empirisches Scaling, könnte anders sein
   - GEFAHR: Quadratisches Scaling als Dogma statt Hypothese

## 5. Empfohlener erster Umbau-Slice

### Priorität 1: SSM/res/01_cli_precheck.py
```python
# Vollständiger Umbau analog zu analyze_wang/01 Struktur:
- Args konsistent machen (M, tau/years, start_idx)
- Print-Format von Wang übernehmen (klare Sections)
- SSM_res_data.load_data() nutzen statt direkter ssmlearn_res
- Eigenwert-Tabelle mit Perioden in Jahren
```

### Priorität 2: SSM/res/03_cli_phase_lock.py → umbenennen zu 03_cli_phase.py
```python
# Fixes:
- Default ssm_dim=4
- Konsistente Imports aus SSM_res_*
- 4-Panel-Plot analog Wang (envelopes, phases, delta, polar)
- phase_bandwidth Parameter hinzufügen
```

### Priorität 3: SSM/res/05_cli_scan.py
```python
# Kompletter Rewrite nötig:
- Lokale Scan-Funktion implementieren
- Tabellen-Format von Wang übernehmen
- 4-Panel Summary Plot
- Keine analyze_residuals Dependencies
```

## Technische Empfehlungen

1. **Module-Struktur beibehalten**:
   - SSM_res_data.py (BTC-spezifisch mit Halvings, Cycle-Tops)
   - SSM_res_embedding.py (generisch, wiederverwendbar)
   - SSM_res_phase.py (neu erstellen für Phase-Lock-Funktionen)

2. **Konsistente Parameter**:
   - M=35, tau=41 (oder years=3.77), start_idx=1164 als Defaults
   - ssm_dim=4, poly_degree=1 für Hauptfall

3. **Print-Formate von Wang übernehmen**:
   - Klare "=" Separatoren
   - Sections: SIMULATION/DATA, MODES, HARMONICS, PHASE LOCK, etc.
   - Tabellarische Ausgaben mit festen Spaltenbreiten

4. **Plots konsistent**:
   - Dunkles Theme (hz.mplstyle)
   - 4-Panel-Layouts wo sinnvoll
   - Jahre statt Tage auf X-Achse

5. **Keine Wang-spezifischen Features**:
   - Kein IC-Parameter
   - Kein transient_frac
   - Keine ODE-Parameter (a,b,c,d)
   - Fokus auf empirische Analyse, nicht Simulation