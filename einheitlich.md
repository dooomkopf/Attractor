# Einheitlicher Stand

Dieses Dokument ist die Kurzreferenz fuer Folgesessions. Es soll verhindern,
dass `analyze_wang`, `analyze_residuals`, `SSM/res` und `analyze_LPPL`
wieder durcheinandergeraten.

## Betroffene Verzeichnisse

- `/home/hz/Data/Attractor/analyze_wang`
- `/home/hz/Data/Attractor/analyze_residuals`
- `/home/hz/Data/Attractor/SSM/res`
- `/home/hz/Data/Attractor/analyze_LPPL`
- `/home/hz/Data/Attractor/LPPL-attractor`
- `/home/hz/Data/Attractor/lpplattr02_analysis_common.py`
- `/home/hz/Data/Attractor/SSMToolHaller.md`
- `/home/hz/Data/Attractor/claude_ssmtoolbox_dgl_spec.md`
- `/home/hz/Data/Attractor/DGL.tex`

## Rollen der Zweige

### analyze_wang

Modellbasierter Referenzzweig fuer ein bekanntes DGL-System.

- bekannte ODE
- bekannte Kanaele `x,y,z`
- erst modellbasiert, dann signalnah
- Referenz fuer den Workflow `01..10`

### analyze_residuals

Datengetriebener BTC-Zweig.

- echte BTC-Residualdaten
- TDE -> PCA -> SSMLearn
- hier werden `ssm_dim`, `poly`, Embedding-Parameter untersucht
- keine DGL als Quelle der Wahrheit

### SSM/res

Alte LPPL-/SSM-Vorlage.

- darf als historische Vorlage gelesen werden
- ist nicht mehr automatisch die sauberste Quelle
- wenn etwas unklar ist, nicht blind von hier kopieren

### analyze_LPPL

Neuer modellbasierter LPPL-Zweig.

- soll zum Wang-Zweig analog aufgebaut sein
- muss aber die echte LPPL-Simulationslogik benutzen
- lokale Analyse darf explizit vereinfacht werden, die Simulationsmechanik nicht

## Harte Regel fuer analyze_LPPL

`analyze_LPPL` ist kein freies LPPL-Spielzeug.

Die Simulationsquelle ist:

- `/home/hz/Data/Attractor/LPPL-attractor/lpplattr02.py`
- `/home/hz/Data/Attractor/LPPL-attractor/lpplattr02_ode.py`
- `/home/hz/Data/Attractor/LPPL-attractor/lpplattr02_params.py`
- `/home/hz/Data/Attractor/lpplattr02_analysis_common.py`

Wenn `analyze_LPPL` davon abweicht, muss das explizit und begruendet sein.
Stille Vereinfachungen sind hier verboten.

## Was bei analyze_LPPL identisch bleiben muss

- 3D-LPPL/Wang-DGL mit `y1, y2, z`
- BTC-Tag-Achse aus `ziel.csv`
- keine Kunstzeit `t=1` als Ersatz fuer reale BTC-Tage
- RK8-Butcher-Tableau
- split-step:
  - Sign-Step halber Tag
  - RK8 voller Tag
  - Sign-Step halber Tag
- `SIGN_OU` aktiv
- `sigma_by_cycle` aktiv
- `mu_offset_by_cycle` aktiv
- keine stille Umstellung auf `solve_ivp`

## Was in analyze_LPPL explizit anders sein darf

- lokaler Analysezweig mit `M=1`
  - das ist eine explizite, analysierbare Variante
  - die Originalquelle hat `M ~= 1.071`
  - dieser Unterschied muss im Output ehrlich benannt werden

## Was standardmaessig aus sein soll

- Damping
- Halving-Impulse
- Fork-Impulse

Diese Dinge sind nicht "vergessen", sondern bewusst fuer den
Analyse-Default ausgeschaltet.

## Was standardmaessig an sein soll

- Wang-Kopplung
- `SIGN_OU`
- `sigma_by_cycle`

## Bedeutung von --wang-off

`--wang-off` heisst in `analyze_LPPL`:

- `Z_A = 0`
- `Z_B = 0`
- `Z_MIX = 8e-5`

Default ohne `--wang-off`:

- `Z_MIX = 0.0002`

Wichtig:
- `--wang-off` ist ein Vergleichszweig
- kein neuer Hauptdefault

## CLI-Konventionen

### SSM/res und analyze_residuals

Einzelfall:

- `--ssm_dim INT`
- `--poly INT`

Scan:

- `--scan_ssm_dim CSV`
- `--scan_poly CSV`

Nicht mischen.

### analyze_LPPL

Es gibt zwei verschiedene `M`:

- `--M`
  - Embedding-Dimension fuer SSMLearn/TDE
- `--lppl_M`
  - Exponent der LPPL-DGL

Entsprechend:

- `--lppl_N`
- `--alpha`
- `--gamma`
- `--Z_A`
- `--Z_B`
- `--Z_C`
- `--Z_D`
- `--Z_E`
- `--Z_MIX`

Das ist absichtlich getrennt, damit Embedding-`M` und DGL-`M` nicht kollidieren.

## Warum hier Genauigkeit so wichtig ist

Die wiederkehrenden Fehler waren:

1. lokale Analysevereinfachung mit Simulationsvereinfachung verwechseln
2. Kunstzeit statt echter BTC-Tage benutzen
3. split-step durch generischen ODE-Solver ersetzen
4. `M=1` still so behandeln, als sei die Original-DGL unveraendert
5. `--wang-off` falsch interpretieren
6. Parameter nur in einem Teilpfad durchreichen
7. denselben Workflow-Index `01..10` fuer fachlich verschiedene Dinge benutzen

Das fuehrt sofort zu falschen Aussagen ueber:

- Stabilitaet
- Phase-Lock
- Harmonische
- V2.6-/SSM-Pfaede

## Aktueller Zielzustand

### analyze_wang

Kontrolliertes Referenzsystem:

- modellbasiert
- signalnah
- SSM-Ergaenzung

### analyze_LPPL

LPPL-Pendant zu `analyze_wang`:

- echte LPPL-Simulationsmechanik
- gleiche Workflow-Idee `01..10`
- zusaetzlich lokale SSM-/SSMLearn-Auswertung auf den simulierten Daten

### analyze_residuals

Echte BTC-Daten:

- datengetrieben
- kein Ersatz fuer den modellbasierten LPPL-Zweig

## Plot-Dateien

In `analyze_LPPL` sollen Plot-CLIs ihre PNGs automatisch abspeichern.
Dateiname:

- Prefix `01_`, `02_`, ...
- dazu die wichtigsten Optionen im Namen

## Wenn eine neue Session startet

Immer zuerst lesen:

1. dieses Dokument
2. `/home/hz/Data/Attractor/analyze_LPPL/README.md`
3. die Originalquellen `lpplattr02*.py`

Und dann erst anfassen.
