The review has been saved to `/home/hz/Data/Attractor/SSM_res_review.md`. 

## Key Findings:

**1. Fachliche Fehler in SSM/res:**
- Default `ssm_dim=2` in 03_cli_phase_lock.py (sollte 4 sein)
- Kaputte Import-Pfade in 05_cli_scan.py
- Inkonsistente Kommentare (falsche Nummerierung)

**2. Architektur-Empfehlung:**
- Von analyze_residuals **inspiriert** werden, nicht direkt aufsetzen
- SSM/res eigene Module behalten für BTC-spezifische Features
- Print-Formate und Workflow-Struktur von Wang übernehmen

**3. Kritische Fixes:**
- ssm_dim=4 als Default
- Import-Pfade reparieren
- run_slave_test lokal implementieren

**4. Wang-Analogie-Gefahren:**
- Keine ICs bei BTC (historische Daten)
- Keine Transienten-Entfernung
- Empirische statt exakte 2:1 Resonanz
- Parameter-Scans nur für Analyse, nicht Physik

**5. Erster Umbau:**
- 01_cli_precheck.py komplett analog Wang strukturieren
- 03_cli_phase.py mit korrekten Defaults
- 05_cli_scan.py neu schreiben ohne analyze_residuals Dependencies
