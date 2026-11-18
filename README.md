# photo_cleanup – ähnliche Fotos finden, das beste behalten

> **English:** Local macOS tool that groups near-duplicate shots in your Apple Photos library (SigLIP embeddings + time window), ranks each group with TOPIQ, LAION aesthetics and Apple Vision (aesthetics, face quality, eyes open), and tags the best shot and the deletion candidates with keywords. It never deletes anything. Docs below are in German.

## Fork changes (tslmy)

- **`--library PATH`** picks the `.photoslibrary` explicitly (e.g. one on an external drive). Keywords are only written when the Photos app has that same library open; otherwise tagging is skipped with a message.
- **Scoped tagging:** a run only removes keywords from photos in its own `--since`/`--until` range, so scanning year by year keeps earlier years' tags. Keywords are sent to Photos in batches of 500.
- **New `lowq` command:** flags standalone low-quality photos (low TOPIQ, low Apple aesthetic score, or Apple's "utility" class such as screenshots and documents), optionally confirmed per photo by a local VLM (`--vlm`), writes `~/.photo_cleanup/reports/lowq *.csv` and tags `cleanup_lowq`. Thresholds: `--topiq-max` (0.30), `--aes-max` (-0.30).
- **VLM:** English prompts, `think: false`, JSON parsed from the reply text (Ollama's MLX runner rejects `format: json`), default model `qwen3.5:9b-mlx`.
- **uv-managed:** `uv sync`, then `uv run photo_cleanup.py …`.

```bash
uv run photo_cleanup.py scan --library "/Volumes/<drive>/<name>.photoslibrary" --limit 500 --no-albums
uv run photo_cleanup.py lowq --library "/Volumes/<drive>/<name>.photoslibrary" --limit 500 --no-albums --vlm
```

---

Läuft komplett lokal auf dem Mac. Liest die Fotos-Mediathek, gruppiert ähnliche Aufnahmen, wählt pro Gruppe das beste Foto und markiert es mit dem Schlagwort **bestes_foto**, alle anderen mit **loeschkandidat**. **Es löscht nichts** – du schaust die Kandidaten in Fotos durch und löschst selbst.

## Setup (einmalig)

```bash
cd photo-cleanup
uv sync
```

- **Terminal braucht Zugriff auf Fotos:** Systemeinstellungen → Datenschutz & Sicherheit → *Festplattenvollzugriff* → Terminal (bzw. iTerm) aktivieren.
- Beim ersten Setzen der Schlagwörter fragt macOS, ob Terminal **Fotos steuern** darf → erlauben.
- Einmalig in Fotos: *Ablage → Neues intelligentes Album* → „Schlagwort ist loeschkandidat“. Zum Vergleichen eine zweite Regel „Schlagwort ist bestes_foto“ mit *Beliebige* verknüpfen – dann liegen Gewinner und Kandidaten jeder Serie nebeneinander. Das Album füllt sich bei jedem `scan` selbst. (Fotos per Skript direkt in Alben legen blockiert macOS 27.)
- Beim ersten Lauf werden die Modelle geladen (SigLIP, TOPIQ, LAION, zusammen ca. 1–2 GB).
- Apples Ästhetik-Score braucht **macOS 15+**. Auf älteren Systemen wird er übersprungen.

Optional, für knappe Entscheidungen per Vision-LLM:

```bash
brew install ollama && ollama serve &
ollama pull qwen3.5:9b-mlx
```

## Benutzung

```bash
# Erst klein testen: die neuesten 500 Fotos, nur Bericht
uv run photo_cleanup.py scan --limit 500 --no-albums

# Ein Jahr, mit Schlagwörtern
uv run photo_cleanup.py scan --since 2025-01-01 --until 2025-12-31

# Mit Vision-LLM als Schiedsrichter bei knappen Gruppen
uv run photo_cleanup.py scan --since 2025-01-01 --vlm
```

Die Analyse wird in `~/.photo_cleanup/cache.sqlite` gespeichert. Ein zweiter Lauf analysiert nur neue Fotos und ist deshalb schnell. Berichte (CSV mit allen Scores) liegen in `~/.photo_cleanup/reports/`.

## Ablauf

1. `scan` ausführen
2. In Fotos das intelligente Album (Schlagwort **loeschkandidat**) öffnen, durchsehen, mit ⌘⌫ löschen, was weg soll
3. `uv run photo_cleanup.py learn` – das Skript schaut, welche Kandidaten du wirklich gelöscht hast, und passt die Gewichtung an deinen Geschmack an
4. Beim nächsten `scan` gelten die gelernten Gewichte

## Feintuning

| Option | Standard | Wirkung |
|---|---|---|
| `--sim` | 0.90 | Höher (0.94) = nur fast identische Fotos gruppieren. Niedriger (0.85) = auch „gleiche Szene, anderer Winkel“. Tipp: Die Gruppierung verkettet (A≈B, B≈C ⇒ A, B, C in einer Gruppe). Bei 0.90 entstehen so leicht große Gruppen mit recht unterschiedlichen Fotos; 0.94 ist ein sichererer Start. |
| `--time-gap` | 30 | Max. Sekunden zwischen zwei Fotos derselben Serie |
| `--vlm-margin` | 0.08 | Ab welchem Score-Abstand eine Gruppe als knapp gilt und ans LLM geht |

Ein guter Einstieg: mit `--no-albums` laufen lassen, die CSV in Numbers öffnen, ein paar Gruppen gegenprüfen und `--sim` anpassen.

## Was in den Score einfließt

| Merkmal | Quelle | Standardgewicht |
|---|---|---|
| `topiq` | TOPIQ – technische Qualität (Schärfe, Rauschen, Belichtung) | 0.35 |
| `laion` | LAION Aesthetic Predictor – Ästhetik | 0.20 |
| `apple_aes` | Apple Vision Ästhetik-Score | 0.15 |
| `face_q` | Apple Vision Gesichtsqualität | 0.15 |
| `eyes` | Augenöffnung aus Gesichts-Landmarks (schlechtestes Gesicht zählt) | 0.15 |

Alle Merkmale werden innerhalb jeder Gruppe auf 0–1 normalisiert. Wenn jemand blinzelt, gibt es zusätzlich einen Abzug. Fehlt ein Modell, wird das Merkmal einfach weggelassen.

## Grenzen

- Getestet auf macOS 27 mit Python 3.12. Unter macOS 27 lässt sich per Skript (AppleScript wie PhotoKit) kein Foto in ein Album legen – deshalb Schlagwörter + intelligentes Album.
- Ehrliche Einschätzung aus der Praxis: Durchsehen musst du die Kandidaten trotzdem selbst. Das Tool ist vor allem nützlich, um Serien zu finden und einen Vorschlag zu bekommen – ob das schneller ist als Wischen in der Fotos-App, hängt von deiner Mediathek ab.

- Fotos, die nur in iCloud liegen und keine lokale Vorschau haben, werden übersprungen (die Anzahl steht in der Ausgabe). Abhilfe: in Fotos → Einstellungen → iCloud kurz „Originale auf diesen Mac laden“.
- Die Augen-Erkennung ist eine Heuristik und bei Profilen oder Sonnenbrillen unzuverlässig.
- Videos und Live-Photo-Videoteile werden nicht betrachtet.
