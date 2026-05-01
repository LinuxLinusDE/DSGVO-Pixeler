# AGENTS.md

Hinweise fuer Codex und andere Coding-Agents in diesem Repository.

## Projektueberblick

DSGVO-Pixeler ist ein lokales Python-Tool zur Anonymisierung von Videos. Das Hauptskript `dsgvo-pixeler.py` erkennt Kennzeichen und Gesichter mit YOLOv8 und anonymisiert diese per Blur oder Pixelation. Die Projektseite liegt unter `docs/`.

## Wichtige Dateien

- `dsgvo-pixeler.py`: Hauptprogramm und CLI.
- `requirements.txt`: Python-Abhaengigkeiten.
- `README.md`: Englische Dokumentation.
- `README_DE.md`: Deutsche Dokumentation.
- `docs/index.html`: Deutsche Projektseite.
- `docs/index.en.html`: Englische Projektseite.
- `docs/command-builder.html`: UI zum Zusammenstellen von CLI-Befehlen.
- `models/`: Erwartete Ablage fuer YOLO `.pt` Modellgewichte.

## Entwicklungssetup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Zusaetzlich wird `ffmpeg`/`ffprobe` benoetigt, auf macOS typischerweise via:

```bash
brew install ffmpeg
```

## Lokale Ausfuehrung

Ein minimaler Lauf sieht so aus:

```bash
python dsgvo-pixeler.py --input input.mp4 --output output.mp4 --weights models/plates/best.pt
```

Fuer schnelle Tests bevorzugt kleine Testvideos oder begrenzte Laufzeit nutzen:

```bash
python dsgvo-pixeler.py --input input.mp4 --test_minutes 1 --preset fast
```

## Tests und Verifikation

Es gibt derzeit keine dedizierte Testsuite. Nach Codeaenderungen mindestens pruefen:

```bash
python dsgvo-pixeler.py --help
python -m py_compile dsgvo-pixeler.py
```

Bei funktionalen Aenderungen an der Video-Pipeline zusaetzlich einen kurzen Lauf mit einem kleinen lokalen Video durchfuehren. Falls keine Modellgewichte oder Testvideos vorhanden sind, dies im Abschluss klar nennen.

## Code-Konventionen

- Bestehenden Stil im Skript beibehalten: einfache Funktionen, explizite Argumente, robuste Fallbacks.
- Keine unnoetigen neuen Abstraktionen einfuehren.
- CLI-Optionen in `argparse` dokumentieren und Default-Verhalten klar halten.
- Fehlermeldungen sollen fuer normale Nutzer verstaendlich bleiben.
- ASCII bevorzugen, da viele bestehende Texte bewusst ohne Umlaute geschrieben sind.

## Datenschutz und Artefakte

- Keine privaten Videos, Snapshots, Outputs oder Modellgewichte committen.
- `.pt` Dateien unter `models/` sind lokale Abhaengigkeiten und koennen gross oder lizenzrechtlich sensibel sein.
- Keine Beispielvideos oder personenbezogenen Daten ins Repository aufnehmen.
- Generierte Outputs, temporare Dateien und lokale Caches nicht einchecken.

## Dokumentation

Bei sichtbaren CLI- oder Verhaltensaenderungen die passende Dokumentation aktualisieren:

- Deutsch: `README_DE.md` und `docs/index.html`
- Englisch: `README.md` und `docs/index.en.html`
- Falls Befehlsoptionen betroffen sind: `docs/command-builder.html`

Deutsch und Englisch muessen inhaltlich konsistent bleiben, auch wenn Formulierungen nicht wortgleich sind.

## Frontend-Hinweise fuer `docs/`

- Die Projektseite ist statisches HTML/CSS/JS, ohne Build-Schritt.
- Bestehendes Layout und visuelle Sprache beibehalten.
- Links und Beispiele nach CLI-Aenderungen gegenpruefen.
- Bilder unter `docs/misc/` nur ersetzen, wenn das bewusst Teil der Aufgabe ist.

## Git-Hygiene

- Keine fremden oder unzusammenhaengenden Aenderungen zuruecksetzen.
- Vor groesseren Aenderungen `git status --short` pruefen.
- Scope klein halten und generierte Dateien nur aufnehmen, wenn sie explizit benoetigt werden.
