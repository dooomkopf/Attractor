# SSMTool 2.6 (Jain / Li / Haller) — Code-nahe Workflow-Dokumentation

Quelle des neuen Repos:
- `/home/hz/Data/Attractor/SSMTool_jain/`
- `git log -n 1`: `de6cbc7 SSMTool 2.6`

Vergleichsbasis:
- altes Repo: `/home/hz/Data/Attractor/SSMtool/`
- alte Langdoku: `/home/hz/Data/Attractor/SSMToolHaller.md`
- datengetriebene Gegenfamilie: `/home/hz/Data/Attractor/SSMLearnPy/`

Ziel dieser Datei:
- nicht die alte V1.0-Doku nachzuerzählen,
- sondern den **echten Workflow des neuen Repos** entlang des Codes zu dokumentieren,
- und präzise festzuhalten, was gegenüber SSMtool V1.0/Addendum wirklich neu ist.

---

## 1. Einordnung

`SSMTool_jain` ist weiterhin eine **modellbasierte** SSM-Toolbox.

Es ist **nicht**:
- eine neue SSMLearn-Version,
- keine datengetriebene Manifold-Learning-Pipeline,
- keine reine GUI-Toolbox.

Es ist:
- eine objektorientierte MATLAB-Plattform zur Berechnung von SSMs,
- reduzierter Dynamik,
- Backbone-Kurven,
- FRC/FRS,
- Stabilitätsdiagrammen,
- Torus-/Quasi-Periodik-Workflows,
- mit expliziten Pfaden für interne Resonanzen,
- und mit intrusiven, semi-intrusiven und non-intrusiven Modellrepräsentationen.

Die wichtigste konzeptionelle Änderung gegenüber V1.0 ist:

> Das neue Repo ist eine **allgemeine SSM-Plattform**.  
> Das alte Repo war primär eine **kleine 2D-SSM-Toolbox für mechanische Systeme** mit GUI und Addendum.

---

## 2. Harte Fakten zur Größe und Struktur

| Repo | Größe | Dateien | Charakter |
|---|---:|---:|---|
| `SSMtool` | `16M` | `91` | kleine GUI-zentrierte Toolbox V1.0 plus Addendum |
| `SSMTool_jain` | `273M` | `3363` | große Plattform mit OOP-Kern, vielen Beispielen und vendorten Abhängigkeiten |

Top-Level von `SSMTool_jain`:
- `src/`
- `examples/`
- `ext/`
- `install.m`

Wichtige Teilbäume:
- `src/@DynamicalSystem/`
- `src/@Manifold/`
- `src/@SSM/`
- `src/misc/`
- `ext/coco/`
- `ext/YetAnotherFEcode/`
- `ext/tensor_toolbox/`
- `ext/torus_collocation/`
- `ext/Wrappers/`

Praktische Folge:
- ein großer Teil der Repo-Größe kommt von **COCO**, **FE-Code**, **Tensor-Toolbox** und Beispielen,
- nicht nur vom eigentlichen SSM-Kern.

---

## 3. Kernarchitektur

### 3.1 Die drei Hauptklassen

Der eigentliche Rechenkern sitzt auf drei Klassen:

1. `DynamicalSystem`
   - Datei: `/home/hz/Data/Attractor/SSMTool_jain/src/@DynamicalSystem/DynamicalSystem.m`
   - beschreibt das Vollsystem
   - kann first-order oder second-order sein
   - enthält lineare Teile, Nichtlinearitäten, Forcing und Spektraldaten

2. `Manifold`
   - Datei: `/home/hz/Data/Attractor/SSMTool_jain/src/@Manifold/Manifold.m`
   - verwaltet Spektralunterraum `E`, Resonanzdaten und die eigentliche Mannigfaltigkeitsberechnung
   - Hauptmethoden:
     - `choose_E(...)`
     - `compute_whisker(...)`
     - `compute_perturbed_whisker(...)`
     - `compute_auto_invariance_error(...)`
     - `compute_analyticity_domain(...)`

3. `SSM`
   - Datei: `/home/hz/Data/Attractor/SSMTool_jain/src/@SSM/SSM.m`
   - erbt von `Manifold`
   - kapselt Anwendungs-Workflows:
     - `extract_backbone(...)`
     - `extract_FRC(...)`
     - `extract_FRS(...)`
     - `extract_ridges_trenches(...)`
     - `extract_Stability_Diagram(...)`
     - diverse Continuation-/Sweep-Routinen

### 3.2 Options-Klassen

Das Repo hat explizite Options-Objekte:
- `/home/hz/Data/Attractor/SSMTool_jain/src/DSOptions.m`
- `/home/hz/Data/Attractor/SSMTool_jain/src/ManifoldOptions.m`
- `/home/hz/Data/Attractor/SSMTool_jain/src/FRCOptions.m`
- `/home/hz/Data/Attractor/SSMTool_jain/src/FRSOptions.m`

Das ist ein echter Unterschied zu V1.0:
- nicht mehr GUI-States und verstreute Flags,
- sondern programmatische Konfiguration.

---

## 4. Welche Systemtypen das neue Repo tatsächlich unterstützt

### 4.1 First-order und second-order

`DynamicalSystem` unterstützt explizit beide Fälle:

- second-order:
  - `M xdd + C xd + K x + fnl(x,xd) = fext(t)`
- first-order:
  - `B zdot = F(z) + Fext(t)`

Beleg:
- Kommentarblock in `/home/hz/Data/Attractor/SSMTool_jain/src/@DynamicalSystem/DynamicalSystem.m`

Das ist eine klare Erweiterung gegenüber V1.0, das methodisch und dokumentativ stark second-order/mechanical zentriert war.

### 4.2 Drei Nichtlinearitätsmodi

`DynamicalSystem` trennt:

- intrusive
  - `fnl`, `dfnl`, `F`, `dF`
- semi-intrusive
  - `fnl_semi`, `dfnl_semi`, `F_semi`, `dF_semi`
- non-intrusive
  - `fnl_non`, `dfnl_non`, `F_non`, `dF_non`

Das ist wichtig:
- `non-intrusive` heißt hier **nicht** datengetrieben,
- sondern weiter modellbasiert, nur ohne klassische explizite Tensoreingabe.

### 4.3 Externe Forcierung

Forcing wird über `DynamicalSystem.add_forcing(...)` eingebracht. Das sieht man in vielen Beispielen:
- `/home/hz/Data/Attractor/SSMTool_jain/examples/OscillatorChain/OscillatorChain.m`
- `/home/hz/Data/Attractor/SSMTool_jain/examples/BernoulliBeam/BernoulliBeam.m`

Die nichtautonomen Workflows bauen darauf auf.

---

## 5. Installation und tatsächlicher Nutzungsstil

Installation:
- `/home/hz/Data/Attractor/SSMTool_jain/install.m`

`install.m` macht im Kern:
- `run ext/coco/startup.m`
- `addpath` für `combinator`
- `addpath` für `tensor_toolbox`
- `addpath` für `YetAnotherFEcode`
- `addpath` für `torus_collocation`
- `addpath` für `ext/Wrappers`
- `addpath` für `src` und `src/misc`

Der typische Nutzungsstil in den Beispielen ist:

1. `run ../../install.m`
2. Modell erzeugen
3. `DS = DynamicalSystem(DSorder);`
4. Matrizen/Nichtlinearität/Forcing setzen
5. `DS.linear_spectral_analysis();`
6. `S = SSM(DS);`
7. `S.choose_E(masterModes);`
8. autonom oder nichtautonom weiterrechnen

Belege:
- `/home/hz/Data/Attractor/SSMTool_jain/examples/OscillatorChain/OscillatorChain.m`
- `/home/hz/Data/Attractor/SSMTool_jain/examples/BernoulliBeam/BernoulliBeam.m`
- `/home/hz/Data/Attractor/SSMTool_jain/examples/NACAWing/NACAWing_NonIntrusive.m`

### 5.1 Kanonischer API-Workflow auf einer Seite

Der kleinste echte Nutzungsablauf des neuen Repos ist:

```matlab
run ../../install.m

DS = DynamicalSystem(DSorder);
set(DS, ...);                 % M/C/K/fnl oder A/B/F oder non-intrusive Handles
DS.add_forcing(...);          % optional

[V,D,W] = DS.linear_spectral_analysis();

S = SSM(DS);
set(S.Options, ...);
S.choose_E(masterModes);

[W_0,R_0] = S.compute_whisker(order);                 % autonom
[W_1,R_1] = S.compute_perturbed_whisker(order-1,...); % nichtautonom

BB  = S.extract_backbone(masterModes,omegaRange,order);
FRC = S.extract_FRC('freq',omegaRange,order);
```

Die wichtigste Umstellung gegenüber V1.0 ist genau hier sichtbar:
- kein GUI-Button-Workflow,
- sondern ein expliziter API-Pfad mit Objekten und Methoden.

---

## 6. Der autonome Kernworkflow

### 6.1 System aufbauen

Beispiele liefern meist ein `build_model.m`, etwa:
- `/home/hz/Data/Attractor/SSMTool_jain/examples/TwoOscillators/build_model.m`
- `/home/hz/Data/Attractor/SSMTool_jain/examples/CharneyDeVore1stOrder/build_model.m`

Danach:
- `DS = DynamicalSystem(DSorder);`
- `set(DS, ...)`

Beispiele:
- second-order mechanisch:
  - `set(DS,'M',M,'C',C,'K',K,'fnl',fnl);`
- non-intrusive:
  - `set(DS,'M',M,'C',C,'K',K,'fnl_non',fnl);`
  - `set(DS.Options,'Intrusion','none')`

### 6.2 Spektralanalyse

Lineare Analyse läuft über:
- `/home/hz/Data/Attractor/SSMTool_jain/src/@DynamicalSystem/linear_spectral_analysis.m`

Wichtige Punkte dort:
- kleine Systeme: `eig(A)` oder `eig(A,B)`
- große Systeme: `eigs(...)`
- Sortierung nach Realteil / Imaginärteil
- positive Imaginärteile werden in komplexen Paaren zuerst angeordnet
- Normierung von `V` und `W`
- Spektrum wird in `obj.spectrum` gespeichert

Das ist wesentlich systematischer als in V1.0.

### 6.3 Masterraumwahl und Resonanzanalyse

Masterraumwahl:
- `/home/hz/Data/Attractor/SSMTool_jain/src/@Manifold/choose_E.m`

`choose_E(...)` macht zwei Dinge:
- setzt `obj.E.spectrum`, `obj.E.basis`, `obj.E.adjointBasis`
- führt `resonance_analysis(...)` aus

Die Resonanzanalyse liefert:
- `resonance.outer`
- `resonance.inner`
- `sigma_out`
- `sigma_in`

Das ist stärker formalisiert als in V1.0:
- Resonanzprüfung ist jetzt ein explizites Objekt des Workflows,
- nicht nur ein Randcheck vor der SSM-Berechnung.

### 6.4 Autonome SSM-Berechnung

Autonomer Kern:
- `/home/hz/Data/Attractor/SSMTool_jain/src/@Manifold/compute_whisker.m`

Logik:
- lineare Startterme:
  - `Lambda_E = obj.E.spectrum`
  - `W_01 = obj.E.basis`
- danach ordnungsweise Rekursion:
  - `cohomological_solution(...)`

Repräsentationen:
- `tensor`
- `multiindex`

Das ist ein großer Unterschied zu V1.0:
- der neue Code unterstützt mehrere interne Darstellungsformen,
- V1.0 war viel direkter und enger verdrahtet.

### 6.5 Backbone-Kurven

Backbone-Workflow:
- `/home/hz/Data/Attractor/SSMTool_jain/src/@SSM/extract_backbone.m`

Tatsächliche Reihenfolge:
1. `obj.choose_E(modes)`
2. `obj.compute_whisker(order(end))`
3. `compute_gamma(R0)`
4. `frc_ab(...)`
5. `compute_output_polar2D(...)`
6. `plot_FRC(...)`

Einschränkung:
- analytische Backbone-Berechnung ist auf **2D-SSMs** ausgelegt
- `assert(numel(modes)==2, ...)`

Das ist ein wichtiger Punkt:
- obwohl das Repo insgesamt höherdimensionale resonante Fälle kann,
- bleibt der klassische analytische Backbone-Pfad eng an 2D gebunden.

---

## 7. Der nichtautonome Kernworkflow

### 7.1 Zentrale Funktion

Nichtautonomer Kern:
- `/home/hz/Data/Attractor/SSMTool_jain/src/@Manifold/compute_perturbed_whisker.m`

Diese Funktion berechnet:
- `W1`
- `R1`

also:
- nichtautonome SSM-Koeffizienten
- nichtautonome reduzierte Dynamik

Sie ist der eigentliche Nachfolger des alten Addendum-Denkens, aber jetzt als Kernfunktion des Repos.

### 7.2 Bedeutung

`compute_perturbed_whisker(...)` erweitert den autonomen Fall um:
- periodische oder quasi-periodische Forcierung,
- Fourier-Indizes `kappa`,
- Ordnung in `epsilon`,
- nichtautonome Beiträge in SSM und reduzierter Dynamik.

Das ist kein Add-on mehr, sondern integraler Bestandteil vieler FRC-/FRS-Pfade.

### 7.3 First-order und second-order Varianten

Intern verzweigt der Code nach Systemtyp:
- `nonAut_1stOrder_whisker(...)`
- `nonAut_2ndOrder_whisker(...)`

Damit ist die nichtautonome Infrastruktur nicht mehr nur mechanisches Spezialzubehör.

---

## 8. FRC-Workflow im neuen Repo

### 8.1 Top-Level

Haupteinstieg:
- `/home/hz/Data/Attractor/SSMTool_jain/src/@SSM/extract_FRC.m`

`extract_FRC(...)` macht:
1. lineares Spektrum sicherstellen
2. resonante Eigenwerte im Frequenzbereich finden
3. resonante Moden und interne Resonanzen bestimmen
4. geeigneten SSM konstruieren
5. FRC per Level-Set oder Continuation berechnen
6. Ergebnis in physikalische Koordinaten zurückführen und plotten

### 8.2 Drei Rechenwege

`extract_FRC(...)` verzweigt nach Methode:
- `level set`
- `continuation ep`
- `continuation po`

Kerneinstiege:
- `/home/hz/Data/Attractor/SSMTool_jain/src/@SSM/FRC_level_set.m`
- `/home/hz/Data/Attractor/SSMTool_jain/src/@SSM/FRC_cont_ep.m`
- `/home/hz/Data/Attractor/SSMTool_jain/src/@SSM/FRC_cont_po.m`

### 8.3 Was `FRC_level_set` tatsächlich macht

`FRC_level_set(...)` ist der analytischste FRC-Pfad:

1. `choose_E(resModes)`
2. `compute_whisker(max_order)`
3. autonome Koeffizienten `gamma` und `lambda` bestimmen
4. Gitter in Polar-Koordinaten aufbauen
5. je nach `contribNonAuto`
   - führenden oder höheren nichtautonomen Beitrag via `compute_perturbed_whisker(...)` einbeziehen
6. reduzierte 2D-Polardynamik auswerten
7. Fixpunkte der reduzierten Dynamik bestimmen
8. Stabilität prüfen
9. Antwort zurück in physikalische Koordinaten ausgeben

Das ist deutlich mehr als das alte V1.0-Backbone-Schema.

### 8.4 Rolle interner Resonanzen

`extract_FRC(...)` und die zugehörigen privaten Helfer behandeln interne Resonanzen systematisch:
- resonante Moden erkennen
- `mFreqs` bestimmen
- ggf. höhere SSM-Dimension verwenden

Das ist der zentrale Unterschied zur V1.0-Welt:
- interne Resonanz ist nicht nur Warnung,
- sondern konstitutiver Teil des Workflows.

---

## 9. FRS, ridges/trenches und Stabilitätsdiagramme

### 9.1 Forced Response Surface

Datei:
- `/home/hz/Data/Attractor/SSMTool_jain/src/@SSM/extract_FRS.m`

Workflow:
1. Resonanzkonsistenz prüfen
2. `choose_E(modes)`
3. `compute_whisker(order)`
4. autonome reduzierte Dynamik auf interne Resonanzkonsistenz prüfen
5. führende nichtautonome Beiträge über `compute_perturbed_whisker(0,...)`
6. Datenstruktur für reduzierte Dynamik aufbauen
7. entweder analytisch oder via 2D-Continuation FRS berechnen

Wichtig:
- Das ist nicht nur “mehr FRC”.
- Es ist eine echte 2-Parameter-Oberfläche.

### 9.2 Ridges und Trenches

Datei:
- `/home/hz/Data/Attractor/SSMTool_jain/src/@SSM/extract_ridges_trenches.m`

Workflow:
1. Resonanzprüfung
2. `choose_E(...)`
3. `compute_whisker(order)`
4. nichtautonome führende Beiträge berechnen
5. COCO-kompatibles reduziertes Vektorfeld konstruieren
6. Optimierungs-/Amplitude-Objekt definieren
7. Continuation über Damped Backbone / Resonanzrücken
8. Ergebnisse zurück in den Vollraum mappen

Das existiert im alten Repo so nicht.

### 9.3 Stabilitätsdiagramme

Datei:
- `/home/hz/Data/Attractor/SSMTool_jain/src/@SSM/extract_Stability_Diagram.m`

Workflow:
1. `choose_E(resModes)`
2. `compute_whisker(order)`
3. je nach Einstellung:
   - nichtautonome ROM-Abhängigkeit direkt oder sensitivitätsbasiert behandeln
4. COCO-Problem aufsetzen
5. Bifurkationen wie `PD` oder `SN` detektieren und fortsetzen

Das ist weit jenseits dessen, was V1.0 praktisch als Hauptworkflow bot.

---

## 10. Continuation, COCO und externe Abhängigkeiten

### 10.1 COCO ist Kerninfrastruktur

Das neue Repo nutzt COCO nicht bloß randständig:
- `ext/coco/` ist vendorisiert
- viele FRC-/FRS-/Stability-/Torus-Pfade bauen darauf auf

Beispiele:
- `SSM_epSweeps`
- `SSM_poSweeps`
- `SSM_lvlSweeps`
- `SSM_isol2ep`
- `SSM_isol2po`
- `SSM_TR2tor`

### 10.2 Weitere große Abhängigkeiten

Wichtige externe Pakete:
- `YetAnotherFEcode`
- `tensor_toolbox`
- `torus_collocation`
- `Wrappers`

Praktische Folge:
- Die Repo-Größe ist nicht direkt mit “Kernlogik” gleichzusetzen.
- Ein großer Teil ist Infrastruktur für FE, Continuation und Hochdimensionalität.

---

## 11. Diagnostik und Gültigkeitsprüfung

Das neue Repo hat eigene Prüfpfade, die V1.0 in dieser Form nicht als klaren Workflowblock hatte:

- `/home/hz/Data/Attractor/SSMTool_jain/src/@Manifold/compute_auto_invariance_error.m`
- `/home/hz/Data/Attractor/SSMTool_jain/src/@Manifold/compute_analyticity_domain.m`
- `/home/hz/Data/Attractor/SSMTool_jain/src/@Manifold/compuate_invariance_residual.m`
- `/home/hz/Data/Attractor/SSMTool_jain/src/@Manifold/compute_sensitivity_coefficients.m`

Praktisch bedeuten diese Funktionen:
- Invarianzfehler der berechneten SSM explizit auswerten
- Gültigkeits-/Analytizitätsbereich abschätzen
- Residuen der Invarianzgleichung berechnen
- Sensitivität der reduzierten Dynamik und SSM-Koeffizienten bestimmen

Für unsere lokale Arbeit ist das einer der wertvollsten Unterschiede zum alten Repo:
- nicht nur `W` und `R` berechnen,
- sondern deren Vertrauensbereich und Fehlermaß explizit denken.

## 12. Outputs und wie man sie lesen muss

Die wichtigsten Output-Familien des neuen Repos sind:

- autonome SSM und ROM:
  - `W_0`, `R_0`
- nichtautonome SSM und ROM:
  - `W_1`, `R_1`
- Backbone:
  - `BB`
- Forced Response Curves:
  - `FRC`
- Forced Response Surfaces:
  - FRS-Daten über `extract_FRS`
- ridges/trenches:
  - über `extract_ridges_trenches`
- Stabilitätsdiagramme:
  - `SD`

Wichtig für die Interpretation:
- `compute_whisker` und `compute_perturbed_whisker` liefern die **Kernkoeffizienten**
- `extract_*`-Methoden liefern bereits **anwendungsnahe Auswertungen**
- viele dieser Outputs werden zusätzlich in physikalische Koordinaten zurückgemappt

Die Ebene ist also klar getrennt:
- erst Mannigfaltigkeit und reduzierte Dynamik,
- dann beobachtbare Kurven, Flächen, Stabilitätsgrenzen.

## 13. Appendix A: Was gegenüber V1.0 wirklich neu ist

Die harte Delta-Liste:

1. **Objektmodell statt GUI-zentrierter Hauptfunktion**
   - alt: `compute_SSM.m`
   - neu: `DynamicalSystem` + `Manifold` + `SSM`

2. **First-order-Systeme zusätzlich zu second-order**

3. **Nichtautonome SSMs als Kernworkflow**
   - nicht mehr nur Addendum-Charakter

4. **Interne Resonanzen als systematischer Hauptpfad**

5. **Forced Response Surfaces**
   - nicht nur FRC

6. **Ridges/Trenches**

7. **Stabilitätsdiagramme / Ince-Strutt-artige Workflows**

8. **Torus-/Quasi-Periodik-Workflows**

9. **Nicht-intrusive und semi-intrusive Modellpfade**

10. **Explizite Invarianz- und Analytizitätsdiagnostik**
   - `compute_auto_invariance_error`
   - `compute_analyticity_domain`

11. **Große FE-/COMSOL-orientierte Infrastruktur**

Kurz:

> `SSMTool_jain` ist keine V1.1, sondern eher eine Plattform, in der V1.0 nur noch ein Spezialfall ist.

---

## 14. Was davon eine echte Entsprechung zu SSMLearn ist

Fast nichts direkt.

`SSMLearnPy` arbeitet so:
- Daten
- Embedding
- reduzierte Koordinaten / Manifold Learning
- reduzierte Dynamik-Fits

Beleg:
- `/home/hz/Data/Attractor/SSMLearnPy/README.md`

`SSMTool_jain` arbeitet so:
- bekanntes Modell
- lineare Spektralanalyse
- Masterraumwahl
- Parametrization Method
- analytische / semi-analytische reduzierte Dynamik

Auch `non-intrusive` im neuen Repo ist **nicht** SSMLearn-äquivalent:
- es bleibt modellbasiert,
- nur die Nichtlinearität muss nicht als expliziter Polynomtensor vorliegen.

Deshalb:
- `SSMTool_jain` ist eine neue **modellbasierte** Generation,
- nicht die neue **datengetriebene** Generation.

---

## 15. Limitationen und Transfergrenzen für unsere lokale Arbeit

Für BTC / Residuen / Ensemble-`n` ist `SSMTool_jain` nicht direkt ein Ersatz.

Direkt übertragbar sind vor allem:
- saubere Vorprüfungen
- Resonanzlogik
- explizite Trennung autonom / nichtautonom
- Qualitätsdiagnostik
- klarerer Workflow zwischen Mannigfaltigkeit und reduzierter Dynamik

Nicht direkt übertragbar:
- mechanischer Inputstil
- FE-/COMSOL-Wrapper
- COCO-basierte FRS-/Torus-Workflows

Für unseren Workflow heißt das:
- `SSMTool_jain` ist methodische Referenz,
- `SSMLearnPy` bleibt für datengetriebene BTC-Analysen die richtige Familie.

---

## 16. Empfehlung für die weitere Dokumentation

Wenn diese Datei weiter ausgebaut wird, dann in genau dieser Reihenfolge:

1. `DynamicalSystem` vollständig dokumentieren
2. `Manifold.choose_E` und Resonanzanalyse vollständig dokumentieren
3. `compute_whisker` ordnungsweise dokumentieren
4. `compute_perturbed_whisker` und Fourier-/Kappa-Struktur dokumentieren
5. `extract_backbone`
6. `extract_FRC` mit den drei Methoden
7. `extract_FRS`
8. `extract_ridges_trenches`
9. `extract_Stability_Diagram`
10. danach erst Beispielverzeichnisse im Detail

Das ist der echte Kern des neuen Repos.
