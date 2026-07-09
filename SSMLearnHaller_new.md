# SSMLearn — gibt es hier überhaupt eine neue Entsprechung?

Kurzantwort:
- **Nein, nicht im gleichen Sinn wie bei `SSMTool_jain`.**
- Das neue Repo `/home/hz/Data/Attractor/SSMTool_jain/` ist **kein neues SSMLearn**.

Vergleichsbasis:
- bestehende Alt-Doku: `/home/hz/Data/Attractor/SSMLearnHaller.md`
- Python-Implementierung: `/home/hz/Data/Attractor/SSMLearnPy/`
- neues modellbasiertes Repo: `/home/hz/Data/Attractor/SSMTool_jain/`

Diese Datei dokumentiert deshalb bewusst keine künstliche 1:1-Spiegelung, sondern die Frage:

> Gibt es im neuen SSMTool-Repo überhaupt etwas, das methodisch eine neue SSMLearn-Entsprechung wäre?

---

## 1. Was SSMLearn weiterhin ist

SSMLearn / SSMLearnPy bleibt lokal die datengetriebene Familie:
- Input: Trajektorien von Observablen
- Workflow:
  1. Embedding
  2. Parametrisierung / reduzierte Koordinaten
  3. reduzierte Dynamik
- keine Pflicht, die zugrunde liegende DGL analytisch zu kennen

Beleg:
- [`SSMLearnPy/README.md`](/home/hz/Data/Attractor/SSMLearnPy/README.md)

---

## 2. Was `SSMTool_jain` nicht ist

`SSMTool_jain` ist trotz neuer Fähigkeiten **kein SSMLearn-Nachfolger**.

Warum nicht:
- es baut ein `DynamicalSystem`-Objekt auf einem bekannten Modell auf
- es startet von Spektralanalyse der Linearisierung
- es löst cohomologische Gleichungen auf Modellniveau
- es arbeitet weiter mit Parametrization Method, nicht mit Manifold Learning auf Daten

Zentrale Klassen:
- [`DynamicalSystem.m`](/home/hz/Data/Attractor/SSMTool_jain/src/@DynamicalSystem/DynamicalSystem.m)
- [`Manifold.m`](/home/hz/Data/Attractor/SSMTool_jain/src/@Manifold/Manifold.m)
- [`SSM.m`](/home/hz/Data/Attractor/SSMTool_jain/src/@SSM/SSM.m)

Das ist architektonisch fundamental anders als SSMLearn.

---

## 3. Der scheinbar ähnliche Teil: non-intrusive

Das neue Repo spricht von:
- `non-intrusive`
- `data-free non-intrusive model reduction`

Das klingt oberflächlich SSMLearn-nah, ist es aber nicht.

Gemeint ist hier:
- man gibt keine expliziten Polynomtensoren vor,
- aber man hat weiterhin ein **modellbasiertes** System bzw. eine auswertbare Nichtlinearität/FE-Blackbox,
- aus der SSM-Koeffizienten modellbasiert berechnet werden.

Belege:
- README: “data-free non-intrusive model reduction”
- [`fnl_nonIntrusive.m`](/home/hz/Data/Attractor/SSMTool_jain/src/@Manifold/private/fnl_nonIntrusive.m)
- [`dfnl_nonIntrusive.m`](/home/hz/Data/Attractor/SSMTool_jain/src/@Manifold/private/dfnl_nonIntrusive.m)

Das ist **nicht**:
- Daten -> PCA/Embedding -> Decoder/Encoder -> Regression

Also:
- **non-intrusive** hier ist nicht dasselbe wie **data-driven** in SSMLearn.

---

## 4. Was aus `SSMTool_jain` trotzdem für SSMLearn-artige Workflows nützlich ist

Keine direkte Pipeline-Übernahme, aber nützliche Ideen:

### 4.1 Vorprüfungen

Sinnvoll übernehmbar:
- Spektrum prüfen
- interne Resonanzen prüfen
- Invarianzfehler / Gültigkeitsbereich denken

Neue hilfreiche Funktionen:
- [`compute_auto_invariance_error.m`](/home/hz/Data/Attractor/SSMTool_jain/src/@Manifold/compute_auto_invariance_error.m)
- [`compute_analyticity_domain.m`](/home/hz/Data/Attractor/SSMTool_jain/src/@Manifold/compute_analyticity_domain.m)
- [`detect_resonant_modes.m`](/home/hz/Data/Attractor/SSMTool_jain/src/@SSM/private/detect_resonant_modes.m)

### 4.2 Resonanzorientierter Workflow

Für unsere `harmonic*`-Arbeit relevant:
- Resonanz nicht nur visuell behaupten
- Mastermoden explizit auswählen
- interne Resonanzbeziehungen systematisch prüfen

### 4.3 Saubere Trennung autonom / nichtautonom

Sehr nützlich für spätere Struktur:
- autonomer Teil: `compute_whisker`
- nichtautonomer Teil: `compute_perturbed_whisker`

Für BTC-/Residuenarbeit heißt das nicht “nutze SSMTool direkt”, sondern:
- strukturiere den Workflow sauberer
- trenne freie Dynamik, erzwungene Beiträge und Beobachtungsmodell klarer

---

## 5. Was lokal wirklich die neue SSMLearn-Entsprechung wäre

Wenn wir nach einer “neuen SSMLearn-Entsprechung” suchen, dann liegt sie lokal eher in:
- `SSMLearnPy`
- unserem modularen Umbau in `analyze_residuals/`
- später `analyze_n_ens/`

Nicht in `SSMTool_jain`.

Die richtige Lesart ist also:

| Familie | Lokal |
|---|---|
| klassische modellbasierte SSM-Toolbox alt | `/home/hz/Data/Attractor/SSMtool/` |
| neue modellbasierte SSM-Toolbox | `/home/hz/Data/Attractor/SSMTool_jain/` |
| datengetriebene SSM-Familie | `/home/hz/Data/Attractor/SSMLearnPy/` |
| unser BTC-spezifischer Datenworkflow | `/home/hz/Data/Attractor/analyze_residuals/` |

---

## 6. Konsequenz für unsere Doku

Für `SSMLearnHaller_new.md` gibt es deshalb **keine** echte Vollentsprechung im Stil:
- “hier ist das neue SSMLearn-Repo und so läuft es jetzt”

Die belastbare Aussage ist stattdessen:
- `SSMLearnHaller.md` bleibt die richtige Grundreferenz für datengetriebene Workflows
- `SSMTool_jain` ergänzt die modellbasierte Seite stark
- methodische Ideen aus `SSMTool_jain` können in unsere SSMLearn-/BTC-Pipeline einfließen
- aber das ist **kein** Repo-Wechsel von SSMLearn auf etwas Neues

---

## 7. Empfehlung

Für die weitere lokale Arbeit:
1. `SSMLearnHaller.md` als Datenseite behalten
2. `SSMToolHaller_new.md` als neue Modellseite ausbauen
3. in `analyze_residuals/` gezielt nur diese Dinge aus `SSMTool_jain` übernehmen:
   - Vorprüfungen
   - Resonanzlogik
   - Qualitätsdiagnostik
   - klare Workflow-Trennung

Alles andere würde derzeit nur Begriffe vermischen.
