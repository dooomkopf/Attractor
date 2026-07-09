# Einheitliche BTC-CLI-Oberflaeche

Stand: 2026-04-12

Dieses Dokument haelt die vereinheitlichte CLI-Oberflaeche fuer den BTC-Residual-Zweig fest. Es ist als Uebergabedokument gedacht, damit spaetere Weiterarbeit ohne erneute Rekonstruktion des Problems moeglich ist.

## Kernproblem

Im BTC-Residual-Zweig war die Benutzeroberflaeche inkonsistent:

- Einzelfaelle nutzten teils `--poly_degree`, spaeter `--poly`.
- Scans nutzten teils `--dims`/`--polys`, teils `--scan_ssm_dim`/`--scan_poly`.
- Outputs und Help-Texte verwendeten oft noch alte Namen, obwohl die echten CLI-Flags schon anders hiessen.
- Einige Skripte liefen mit dem neuen Default `--ssm_dim 2 --poly 2` nicht sauber, sondern endeten mit Stacktrace statt mit einem kontrollierten Hinweis.
- `analyze_residuals/05_cli_scan.py` zaehlte oszillatorische Paare falsch, weil dort positive und negative Imaginaerteile doppelt gewertet wurden.
- `05_cli_scan.py` akzeptierte halbe Scan-Aufrufe, z. B. nur `--scan_ssm_dim`, und fuellte das zweite Flag still mit Defaults auf. Genau diese Mischoberflaeche sollte weg.

Das Ergebnis war eine fuer den Nutzer schwer merkbare, teilweise widerspruechliche Oberflaeche.

## Zielregel

Die Benutzeroberflaeche soll strikt so aussehen:

### Einzelfall

- `--ssm_dim INT`
- `--poly INT`

### Scan

- `--scan_ssm_dim CSV`
- `--scan_poly CSV`

### Nicht erlaubt

- keine Mischung aus Einzelfall- und Scan-Flags
- keine sichtbaren Altbegriffe wie `--poly_degree`, `--dims`, `--polys`
- keine Outputs, die andere Namen anzeigen als die echten CLI-Optionen

Wichtig:

- Interne Python-Funktionsparameter dürfen weiterhin `poly_degree` heissen.
- Entscheidend ist die **sichtbare** CLI-Oberflaeche und der **sichtbare** Report.

## Betroffene Verzeichnisse

### Aktiv angepasst

- [/home/hz/Data/Attractor/SSM/res](/home/hz/Data/Attractor/SSM/res)
- [/home/hz/Data/Attractor/analyze_residuals](/home/hz/Data/Attractor/analyze_residuals)

### Nicht Teil dieser letzten Vereinheitlichung

- [/home/hz/Data/Attractor/analyze_wang](/home/hz/Data/Attractor/analyze_wang)

`analyze_wang` wurde in diesem Schritt **nicht** auf dieselbe `--poly`/`--scan_*`-Regel umgebaut, weil dort diese konkrete Inkonsistenz nicht der aktuelle Blocker war.

## Aktive BTC-Skripte in `SSM/res`

- [01_cli_precheck.py](/home/hz/Data/Attractor/SSM/res/01_cli_precheck.py)
- [02_cli_harmonics.py](/home/hz/Data/Attractor/SSM/res/02_cli_harmonics.py)
- [03_cli_phase.py](/home/hz/Data/Attractor/SSM/res/03_cli_phase.py)
- [04_cli_scaling.py](/home/hz/Data/Attractor/SSM/res/04_cli_scaling.py)
- [05_cli_scan.py](/home/hz/Data/Attractor/SSM/res/05_cli_scan.py)
- [06_cli_ssm_system.py](/home/hz/Data/Attractor/SSM/res/06_cli_ssm_system.py)
- [07_cli_ssm_spectral.py](/home/hz/Data/Attractor/SSM/res/07_cli_ssm_spectral.py)
- [08_cli_ssm_whisker.py](/home/hz/Data/Attractor/SSM/res/08_cli_ssm_whisker.py)
- [09_cli_ssm_backbone.py](/home/hz/Data/Attractor/SSM/res/09_cli_ssm_backbone.py)
- [10_cli_ssm_learn.py](/home/hz/Data/Attractor/SSM/res/10_cli_ssm_learn.py)

## Gespiegelte modulare BTC-Skripte in `analyze_residuals`

- [01_cli_precheck.py](/home/hz/Data/Attractor/analyze_residuals/01_cli_precheck.py)
- [02_cli_harmonics.py](/home/hz/Data/Attractor/analyze_residuals/02_cli_harmonics.py)
- [03_cli_phase.py](/home/hz/Data/Attractor/analyze_residuals/03_cli_phase.py)
- [04_cli_scaling.py](/home/hz/Data/Attractor/analyze_residuals/04_cli_scaling.py)
- [05_cli_scan.py](/home/hz/Data/Attractor/analyze_residuals/05_cli_scan.py)
- [10_cli_ssm_learn.py](/home/hz/Data/Attractor/analyze_residuals/10_cli_ssm_learn.py)

## Aktuelle CLI-Regeln

### 1. Precheck

Beispiel:

```bash
cd /home/hz/Data/Attractor/SSM/res
./01_cli_precheck.py
./01_cli_precheck.py --ssm_dim 4 --poly 1
./01_cli_precheck.py --scan_ssm_dim 2,3,4 --scan_poly 2
```

Regel:

- Einzelfall: `--ssm_dim`, `--poly`
- Scan: `--scan_ssm_dim`, `--scan_poly`
- Wenn nur eines der beiden Scan-Flags gesetzt wird, muss das Skript mit `argparse` abbrechen

### 2. Harmonik-Phase

Beispiel:

```bash
cd /home/hz/Data/Attractor/SSM/res
./03_cli_phase.py
./03_cli_phase.py --ssm_dim 4 --poly 1
```

Aktueller Default:

- `--ssm_dim 2`
- `--poly 2`

Das ist **bewusst** ein Single-Mode-Control.

Konsequenz:

- Default beendet sich sauber mit einem Hinweis:
  - keine 2 oszillatorischen Paare
  - fuer echte Harmonikdiagnostik `--ssm_dim 4 --poly 1`

### 3. Harmonik-Scaling

Beispiel:

```bash
cd /home/hz/Data/Attractor/SSM/res
./04_cli_scaling.py
./04_cli_scaling.py --ssm_dim 4 --poly 1
```

Auch hier:

- Default ist Single-Mode-Control
- kein Stacktrace mehr
- klarer Hinweistext statt Absturz

### 4. Harmonik-Scan

Beispiel:

```bash
cd /home/hz/Data/Attractor/SSM/res
./05_cli_scan.py --scan_ssm_dim 2,3,4 --scan_poly 2 --no-show
./05_cli_scan.py --scan_ssm_dim 2,3,4,5 --scan_poly 1,2 --no-show
```

Wichtig:

- `05_cli_scan.py` verwendet **nur** `--scan_ssm_dim` und `--scan_poly`
- keine `--dims`
- keine `--polys`

### 5. Weitere Einzelfall-SSM-Skripte

Beispiele:

```bash
cd /home/hz/Data/Attractor/SSM/res
./07_cli_ssm_spectral.py --ssm_dim 2 --poly 2
./09_cli_ssm_backbone.py --ssm_dim 2 --poly 2 --no-show
./10_cli_ssm_learn.py --ssm_dim 2 --poly 2 --no-show
```

## Default-Interpretation

Der Default fuer die BTC-Residualseite ist aktuell:

```text
--ssm_dim 2
--poly 2
```

Grund:

- `ssm_dim=2` ist der saubere Single-Master-Control
- `poly=2` ist die aktuelle Standard-Polynomordnung fuer den Residuen-Zweig

Wichtige Konsequenz:

- `03_cli_phase.py` und `04_cli_scaling.py` finden im Default **keine** zweite oszillatorische Paarstruktur
- das ist kein Bug, sondern die erwartete Folge des Default-Modells
- fuer die eigentliche Harmonikdiagnostik muss explizit umgeschaltet werden:

```bash
./03_cli_phase.py --ssm_dim 4 --poly 1
./04_cli_scaling.py --ssm_dim 4 --poly 1
```

## Inhaltlicher Befund zum Vergleich `poly=2`, `ssm_dim=2,3,4`

Direkt geprueft wurde:

```bash
cd /home/hz/Data/Attractor/SSM/res
./05_cli_scan.py --scan_ssm_dim 2,3,4 --scan_poly 2 --no-show
```

Ergebnis:

- `s2p2`: 1 Paar, `T_main≈3.841y`
- `s3p2`: 1 Paar, `T_main≈3.977y`
- `s4p2`: 2 Paare, `T_main≈3.807y`, `T_sub≈2.221y`, Detuning `16.7%`, `sub_Re>0`

Interpretation:

- mit `ssm_dim=3` kommt unter `poly=2` nichts grundsaetzlich Neues dazu
- `ssm_dim=4` erzeugt zwar formal ein zweites Paar, aber nicht robust harmonisch
- genau deshalb ist `ssm_dim=2, poly=2` als Default-Control vertretbar

## Was konkret gefixt wurde

### `SSM/res`

- `01`: Output jetzt Wang-aehnlich strukturiert, aber BTC-spezifische Extras am Ende
- `03`: kein Stacktrace mehr im Default-Single-Mode-Fall
- `04`: kein Stacktrace mehr im Default-Single-Mode-Fall
- `05`: strikte Scan-Validierung; halbe Scan-Aufrufe blocken jetzt sauber
- `05`: Scan-Labels auf `s2p2`, `s3p2`, `s4p2`
- `07`, `09`, `10`: sichtbare Reports nutzen `poly`, nicht `poly_degree`

### `analyze_residuals`

- gleiche Oberflaechenregel wie in `SSM/res`
- `03` und `04`: kein Stacktrace mehr im Default-Control
- `05`: positive Imaginaerteile werden als echte Paare gezaehlt, nicht doppelt
- `05`: halbe Scan-Aufrufe blocken jetzt sauber
- `10`: sichtbarer Report auf `poly` umgestellt

## Was erneut komplett getestet wurde

### `SSM/res`

Direkt getestet aus:

```bash
cd /home/hz/Data/Attractor/SSM/res
```

Getestet:

```bash
./01_cli_precheck.py
./02_cli_harmonics.py --no-show
./03_cli_phase.py --no-show
./04_cli_scaling.py --no-show
./05_cli_scan.py --scan_ssm_dim 2,3,4 --scan_poly 2 --no-show
./06_cli_ssm_system.py
./07_cli_ssm_spectral.py --poly 2
./08_cli_ssm_whisker.py
./09_cli_ssm_backbone.py --no-show
./10_cli_ssm_learn.py --no-show
```

### `analyze_residuals`

Direkt getestet aus:

```bash
cd /home/hz/Data/Attractor/analyze_residuals
```

Getestet:

```bash
./01_cli_precheck.py
./02_cli_harmonics.py --no-show
./03_cli_phase.py --no-show
./04_cli_scaling.py --no-show
./05_cli_scan.py --scan_ssm_dim 2,3,4 --scan_poly 2 --no-show
./10_cli_ssm_learn.py --no-show
```

### Zusatztetests fuer die strikte Scan-Regel

Getestet:

```bash
./05_cli_scan.py --scan_ssm_dim 2,3 --no-show
./05_cli_scan.py --scan_poly 2 --no-show
```

Erwartetes und verifiziertes Verhalten:

```text
error: --scan_ssm_dim and --scan_poly must be provided together
```

## Noch offene Themen

### 1. Help-Texte

Die **Funktionalitaet** ist jetzt einheitlich. Es gibt aber noch kosmetische Hilfetext-Reste:

- `argparse` zeigt teilweise noch Metavars wie `POLY_DEGREE`
- das ist sichtbar, obwohl der eigentliche Flag bereits korrekt `--poly` ist

Das ist kein Laufzeitfehler, aber ein sichtbarer Rest, den man noch glattschleifen sollte.

### 2. `analyze_wang`

`analyze_wang` wurde in dieser letzten Oberflaechenrunde **nicht** mitgezogen.

Wenn dort dieselbe Striktheit gewuenscht ist, waere der naechste Schritt:

- `-h` aller aktiven Wang-CLIs pruefen
- sichtbare Metavars und Help-Texte glattschleifen
- aber nur, wenn dort wirklich eine aehnliche Drift vorliegt

### 3. Dokumentation

In einigen `.md`-Dateien stehen noch alte Beispiele mit `--poly_degree` oder alten Scan-Namen.

Wenn die Oberflaeche final stabil ist, sollte man diese Stellen nachziehen, insbesondere:

- `WORKFLOW_BTC_SSM.md`
- `WORKFLOW_OLD_VS_NEW.md`
- evtl. weitere Workflow-Notizen unter `SSM/res`

## Empfohlener naechster Schritt

Wenn spaeter nahtlos weitergemacht werden soll, ist die sinnvolle Reihenfolge:

1. `-h`-Ausgaben der aktiven BTC-CLIs komplett glattschleifen
2. danach die Markdown-Beispiele auf `--poly` und `--scan_*` aktualisieren
3. erst danach entscheiden, ob dieselbe Oberflaechenregel auch fuer `analyze_wang` gelten soll

## Kurzregel fuer kuenftige Aenderungen

Wenn eine BTC-Residual-CLI geaendert wird, gilt:

- Einzelfall:
  - `--ssm_dim`
  - `--poly`
- Scan:
  - `--scan_ssm_dim`
  - `--scan_poly`
- keine anderen sichtbaren Varianten
- Output muss dieselben Namen zeigen wie die CLI selbst

