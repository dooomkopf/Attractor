# CoinGlass: Direkter Datenzugriff statt Hover/OCR

Stand dieser Analyse: 2026-03-29.

## Ziel

Fuer CoinGlass-Metriken wie `Bitcoin Average Block Time` die Rohdaten direkt aus dem laufenden Frontend holen, statt:

- den Chart mit der Maus langsam abzufahren
- ein Video aufzunehmen
- Frames zu extrahieren
- OCR auf Tooltip-Daten zu machen

Das ist fuer historische Vollstaendigkeit und Reproduzierbarkeit deutlich besser.

## Wichtigstes Ergebnis

Ja, es klappt direkt.

Der robusteste Weg ist **nicht**:

- rohe `fetch`-Requests auf die API zu machen
- rohe `curl`-Requests auf die API zu machen
- Tooltips per Maus/Hover zu scrapen

Der robuste Weg ist:

1. CoinGlass im Browser oeffnen.
2. Zur Metrik **ueber die interne Suche** navigieren.
3. Im Seitenkontext CoinGlass' eigenen Frontend-Request-Wrapper verwenden.
4. Dessen bereits eingebaute Request-/Response-Interceptors die Antwort entschluesseln lassen.

## Beobachtete Huerden

### 1. Direkter Seitenaufruf kann unter Automation 404 liefern

Beobachtet wurde:

- `https://www.coinglass.com/de/pro/i/bitcoin-block-time-speed` -> `404 Not Found`
- `https://www.coinglass.com/pro/i/bitcoin-block-time-speed` kann bei direktem Reload/Goto ebenfalls `404` liefern

Aber:

- die Seite ist **intern auf CoinGlass vorhanden**
- Navigation ueber die **interne Suche** funktioniert

Konkreter funktionierender Ablauf:

1. `https://www.coinglass.com/` laden
2. Suchdialog oeffnen
3. nach `block time` suchen
4. `Bitcoin Average Block Time` anklicken

Danach landet man auf:

- `https://www.coinglass.com/pro/i/bitcoin-block-time-speed`

## 2. Rohe API ist nicht direkt nuetzlich

Der Chart zieht Daten aus:

- `https://fapi.coinglass.com/api/metrics/blockSpeedTime`

Was dabei beobachtet wurde:

- Ein nacktes `curl` auf den Endpoint liefert **nicht** die fertige Zeitreihe.
- Ein direktes `fetch(...)` aus der Seite scheitert in dieser Umgebung an CORS.
- Der echte XHR-Response enthaelt zwar Daten, aber im Feld `data` als **verschluesselten/encodierten Blob**.

Das Frontend entschluesselt diese Antwort clientseitig in einem gemeinsamen HTTP-Wrapper.

## Die funktionierende Methode

### Kernidee

CoinGlass hat im Frontend bereits alles, was wir brauchen:

- Request-Wrapper
- Header-/Auth-Handling
- Response-Decryption
- JSON-Normalisierung

Also nicht selbst reverse engineeren, wenn es nicht noetig ist.

Stattdessen im Browser-Kontext:

1. `webpack`-require freilegen
2. das CoinGlass-HTTP-Modul laden
3. dessen Export `FP(...)` direkt aufrufen

## Exakte Schritte

### Schritt 1: Auf der Zielseite landen

Nicht per direktem Reload auf die Metrikseite springen, sondern:

1. CoinGlass Startseite laden
2. Suchdialog oeffnen
3. `block time` eingeben
4. `Bitcoin Average Block Time` anklicken

### Schritt 2: Webpack-Loader greifen

Im Seitenkontext:

```js
let __req;
window.webpackChunk_N_E.push([
  [Math.random()],
  {},
  (req) => {
    __req = req;
  },
]);
```

Danach kann man geladene Frontend-Module direkt per ID laden.

### Schritt 3: CoinGlass' HTTP-Wrapper benutzen

In der verifizierten Sitzung war das relevante Modul:

- `12471`

Aufruf:

```js
const api = __req(12471);
const res = await api.FP({
  url: 'https://fapi.coinglass.com/api/metrics/blockSpeedTime',
});
```

Das Ergebnis ist bereits entschluesselt und normalisiert.

### Schritt 4: Ergebnisform

Verifiziert wurde:

```js
{
  code: "0",
  msg: "success",
  success: true,
  data: [
    { timestamp: 1231459200000, value: 102.86 },
    ...
    { price: 102973.7142276973, timestamp: 1746835200000, value: 9.6 },
    ...
  ]
}
```

Wichtige Beobachtung:

- Fruehe Eintraege haben nur `timestamp` und `value`
- spaetere Eintraege haben zusaetzlich `price`

`timestamp` ist in Millisekunden.

## Vollstaendiges Minimalbeispiel fuer Playwright

```js
await page.goto('https://www.coinglass.com/', { waitUntil: 'domcontentloaded' });

const doNotConsent = page.getByRole('button', { name: 'Do not consent' });
if (await doNotConsent.count()) {
  try {
    await doNotConsent.click({ timeout: 2000 });
  } catch {}
}

await page.getByText('Search /').click();
await page.getByRole('textbox', { name: 'Search coins, metrics,' }).fill('block time');
await page.waitForTimeout(1500);
await page.getByRole('link', { name: 'Bitcoin Average Block Time' }).first().click();
await page.waitForTimeout(4000);

const result = await page.evaluate(async () => {
  let __req;
  window.webpackChunk_N_E.push([
    [Math.random()],
    {},
    (req) => {
      __req = req;
    },
  ]);

  const api = __req(12471);
  const res = await api.FP({
    url: 'https://fapi.coinglass.com/api/metrics/blockSpeedTime',
  });

  return {
    ok: res?.code === '0' && Array.isArray(res?.data),
    count: Array.isArray(res?.data) ? res.data.length : 0,
    first: res?.data?.[0] ?? null,
    last: res?.data?.[res.data.length - 1] ?? null,
    data: res?.data ?? null,
  };
});
```

## Warum das funktioniert

### Relevante Frontend-Module

In der verifizierten Sitzung waren folgende Module relevant:

- `94126`
  - enthaelt die Zuordnung von Metrik-Endpunkten
  - darunter `blockSpeedTime`
- `12471`
  - gemeinsamer HTTP-Wrapper
  - enthaelt Request-/Response-Interceptors
  - entschluesselt Antworten mit verschluesseltem `data`-Feld

### Was der Wrapper intern macht

Die genaue Implementierung ist obfuskiert, aber funktional war klar sichtbar:

- Request-Interceptor setzt notwendige Header
- Response-Interceptor erkennt verschluesselte Payloads
- Payload wird intern entschluesselt
- danach landet fertiges JSON in `response.data`

Zusatzbeobachtung aus dem geladenen Bundle:

- `CryptoJS` ist geladen
- `AES` / `ECB` / `Pkcs7` sind vorhanden
- Kompressions-/Inflate-Code ist ebenfalls geladen

Fuer die Reproduktion ist das aber nur Hintergrundwissen. Die Wrapper-Methode ist einfacher und stabiler.

## Was nicht der bevorzugte Weg ist

### 1. Tooltips per Maus und OCR

Nur als Fallback sinnvoll.

Nachteile:

- langsam
- fehleranfaellig
- Zoom-/Hover-Luecken
- OCR-Artefakte
- schwer reproduzierbar

### 2. Direkter `fetch(...)` im Seitenkontext

In der verifizierten Umgebung schlug der direkte Cross-Origin-Request fehl.

### 3. Direkter `curl` auf den Metric-Endpoint

Ohne die CoinGlass-Frontendlogik bekommt man nicht dieselbe nutzbare Antwort.

## Wenn die Modul-ID `12471` spaeter nicht mehr stimmt

Die ID ist Build-abhaengig und kann sich nach Deployments aendern.

Dann so vorgehen:

1. Seite wie oben ueber interne Suche aufrufen.
2. `__req` wie oben freilegen.
3. Geladene Module durchsuchen.

Beispiel:

```js
let __req;
window.webpackChunk_N_E.push([[Math.random()], {}, (req) => { __req = req; }]);

const hits = [];
for (const [id, fn] of Object.entries(__req.m)) {
  const src = String(fn);
  if (
    src.includes('interceptors.response') ||
    src.includes('blockSpeedTime') ||
    src.includes('mode.ECB') ||
    src.includes('Pkcs7')
  ) {
    hits.push(id);
  }
}
console.log(hits);
```

Heuristik:

- Modul mit `blockSpeedTime` ist meist der Metrik-Registry-Teil
- Modul mit `interceptors.response` ist meist der HTTP-Wrapper

## Empfehlung fuer weitere CoinGlass-Metriken

Das Vorgehen ist sehr wahrscheinlich generalisierbar:

1. Metrikseite ueber CoinGlass-Suche aufrufen
2. `__req` holen
3. HTTP-Wrapper-Modul laden
4. Wrapper mit dem gewuenschten API-Endpoint aufrufen

Wenn der Endpoint unklar ist:

- Netzwerk-Requests der Metrikseite anschauen
- oder das Registry-Modul nach dem Slug durchsuchen

## Empfehlung fuer diesen konkreten Use Case

Fuer `Bitcoin Average Block Time` sollte **nicht** mit Video + OCR weitergearbeitet werden, solange der direkte Datensatz gebraucht wird.

Besser:

1. Daten direkt per Wrapper holen
2. lokal als JSON oder CSV speichern
3. nur fuer visuelle Kontrolle spaeter noch Chart/Hover verwenden

Damit bekommt man:

- alle Datenpunkte
- keine Datums-Luecken
- keine OCR-Fehler
- deutlich bessere Reproduzierbarkeit fuer weitere Automationslaeufe

## Praktischer CSV-Export in dieser Linux-Umgebung

Fuer den reinen Datenzugriff reicht der Wrapper-Aufruf oben.

Fuer den **Dateiexport** in dieser verifizierten Sitzung gab es aber noch eine technische Besonderheit:

- `browser_run_code` konnte in dieser Umgebung **nicht** direkt auf Node-`fs` zugreifen
- `require('fs')` war dort nicht verfuegbar
- dynamisches `import('node:fs/promises')` schlug ebenfalls fehl

Der tatsaechlich verwendete Exportweg war deshalb:

1. Daten per `__req(12471).FP(...)` holen
2. im Seitenkontext auf das gewuenschte Format bringen
3. nach Datum deduplizieren
4. die fertige CSV in mehrere Base64-Chunks zerlegen
5. jeden Chunk per `console.log(...)` mit eindeutigem Tag ausgeben
6. die Browser-Konsole mit `browser_console_messages(..., filename=...)` lokal sichern
7. die Chunks unter Linux per Node wieder zusammensetzen und in die Zieldatei schreiben

### Verwendetes Zielformat

In der verifizierten Sitzung wurden am Ende **zwei** Formate erzeugt:

1. Ein technisches Zwischenformat mit genau drei Zeilen:
   - Zeile 1: alle Blockzeiten
   - Zeile 2: alle Kurse
   - Zeile 3: alle Datumswerte
2. Ein praktisch nutzbares Endformat mit **einer Zeile pro Datenpunkt**:
   - Spalte 1: Blockzeit (`value`, auf 2 Nachkommastellen)
   - Spalte 2: Kurs (`price`, auf 2 Nachkommastellen, sonst `nan`)
   - Spalte 3: Datum (`yyyy.mm.dd`)

Alle Werte sind jeweils durch Leerzeichen getrennt. Es gibt keine Kopfzeile.

### Formatierung und Deduplizierung

Im Seitenkontext wurde effektiv so gearbeitet:

```js
function fmtDate(ts) {
  const d = new Date(ts);
  const yyyy = d.getUTCFullYear();
  const mm = String(d.getUTCMonth() + 1).padStart(2, '0');
  const dd = String(d.getUTCDate()).padStart(2, '0');
  const hh = String(d.getUTCHours()).padStart(2, '0');
  return `${yyyy}.${mm}.${dd}-${hh}`;
}

function fmt2(n) {
  return Number.isFinite(n) ? n.toFixed(2) : 'nan';
}

const seen = new Set();
const deduped = [];

for (const row of rows) {
  const date = fmtDate(row.timestamp);
  if (seen.has(date)) continue;
  seen.add(date);
  deduped.push({
    timebt: fmt2(row.value),
    price: fmt2(row.price),
    date,
  });
}

const content3Rows =
  `${deduped.map((r) => r.timebt).join(' ')}\n` +
  `${deduped.map((r) => r.price).join(' ')}\n` +
  `${deduped.map((r) => r.date).join(' ')}\n`;
```

### Transfer per Browser-Konsole

Beispiel fuer die Ausgabe der Chunks:

```js
const tag = 'CG_EXPORT_BT';
const chunkSize = 6000;
const count = Math.ceil(content.length / chunkSize);

for (let i = 0; i < count; i++) {
  const part = content.slice(i * chunkSize, (i + 1) * chunkSize);
  const b64 = btoa(unescape(encodeURIComponent(part)));
  console.log(`${tag} ${i}/${count} ${b64}`);
}
```

Danach die Konsole lokal speichern und unter Linux wieder zusammensetzen:

```js
const fs = require('fs');
const text = fs.readFileSync('Data/coinglass-console-export.txt', 'utf8');
const tag = 'CG_EXPORT_BT';

const chunks = text
  .split(/\n/)
  .map((line) => line.match(new RegExp(`${tag}\\s+(\\d+)\\/(\\d+)\\s+(.+)\\s+@`)))
  .filter(Boolean)
  .map((m) => ({ i: Number(m[1]), b64: m[3] }))
  .sort((a, b) => a.i - b.i);

const content3Rows = chunks
  .map((c) => Buffer.from(c.b64, 'base64').toString('utf8'))
  .join('');

fs.writeFileSync('Data/ziel_BT_3rows.txt', content3Rows, 'utf8');

const [line1 = '', line2 = '', line3 = ''] = content3Rows.trimEnd().split(/\n/);
const a = line1.split(' ').filter(Boolean);
const b = line2.split(' ').filter(Boolean);
const c = line3.split(' ').filter(Boolean);

const table = c
  .map((date, i) => `${a[i]} ${b[i]} ${date.split('-')[0]}`)
  .join('\n') + '\n';

fs.writeFileSync('ziel_BT.csv', table, 'utf8');
```

### Verifiziertes Ergebnis fuer `blockSpeedTime`

- Rohpunkte: `6288`
- exportierte Punkte nach Datums-Deduplizierung: `6288`
- doppelte Datumswerte im Export: `0`
- erster Tag: `2009.01.09-00`
- letzter Tag: `2026.03.28-00`
- Enddatei `ziel_BT.csv`: `6288` Zeilen mit je `timebt price date`
- Sortierung der Enddatei: chronologisch nach Datum, damit die Zuordnung erhalten bleibt
- Backup `ziel_BT_3rows.txt`: das urspruengliche Drei-Zeilen-Format

Wichtig:

- Doppelte **Blockzeitwerte** sind in dieser CoinGlass-Reihe normal.
- Dedupliziert wurde nur nach formatiertem Datum.
