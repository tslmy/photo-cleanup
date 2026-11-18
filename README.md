# photo_cleanup – find similar photos, keep the best

Runs entirely locally on your Mac. Reads your Apple Photos library, groups similar shots (SigLIP embeddings + time window), picks the best photo in each group (TOPIQ, LAION aesthetics, Apple Vision aesthetics, face quality, eyes open), and tags it with the keyword **cleanup_best** and all others with **cleanup_candidate**. **It never deletes anything** – you review the candidates in Photos and delete them yourself.

> Upgrading from an earlier version? The keywords used to be German (`bestes_foto`, `loeschkandidat`). Old tags are not removed automatically; delete those keywords in Photos (Window → Keyword Manager) and update your smart albums.

## Fork changes (tslmy)

- **`--library PATH`** picks the `.photoslibrary` explicitly (e.g. one on an external drive). Keywords are only written when the Photos app has that same library open; otherwise tagging is skipped with a message.
- **Scoped tagging:** a run only removes keywords from photos in its own `--since`/`--until` range, so scanning year by year keeps earlier years' tags. Keywords are sent to Photos in batches of 500.
- **New `lowq` command:** flags standalone low-quality photos (low TOPIQ, low Apple aesthetic score, or Apple's "utility" class such as screenshots and documents), optionally confirmed per photo by a local VLM (`--vlm`), writes `~/.photo_cleanup/reports/lowq *.csv` and tags `cleanup_lowq`. Thresholds: `--topiq-max` (0.30), `--aes-max` (-0.30).
- **VLM:** English prompts, `think: false`, JSON parsed from the reply text (Ollama's MLX runner rejects `format: json`), default model `qwen3.5:9b-mlx`.
- **uv-managed:** `uv sync`, then `uv run photo_cleanup.py …`.
- **English throughout:** log output, CSV columns, keywords and docs.
- **Review UI (`serve`):** a local web page to go through a report and decide per photo. See [Reviewing in the browser](#reviewing-in-the-browser).

```bash
uv run photo_cleanup.py scan --library "/Volumes/<drive>/<name>.photoslibrary" --limit 500 --no-albums
uv run photo_cleanup.py lowq --library "/Volumes/<drive>/<name>.photoslibrary" --limit 500 --no-albums --vlm
uv run photo_cleanup.py serve --library "/Volumes/<drive>/<name>.photoslibrary"
```

## Reviewing in the browser

```bash
uv run photo_cleanup.py serve [REPORT] [--library PATH] [--port 8765] [--no-open]
```

Opens `http://127.0.0.1:8765/` (local only) with the given report, or the newest one in `~/.photo_cleanup/reports/` if none is given; other reports can be picked from the menu. Both `scan` reports (shown as groups, best shot marked) and `lowq` reports work.

- **Select:** click, ⌘-click to add, ⇧-click for a range, or drag a box over several photos (hold ⌘/⇧ to add to the selection). ⌘A selects everything shown, Esc clears, "Select suggested" picks every photo the tool suggests deleting.
- **Decide:** **D** approve deletion, **K** keep, **R** reset to undecided. Space or double-click opens a large preview (←/→ to browse, D/K/R work there too). Filters show suggested, unsure, undecided, approved or kept photos.
- Decisions are saved as you go to `<report>.decisions.json` next to the report, so you can stop and continue later.
- **Apply approved…** tags exactly the approved photos with the keyword `cleanup_delete` (and removes it from photos in this report you no longer approve). It needs `--library` and refuses to run unless the Photos app has that same library open. Photos doesn't let scripts delete photos, so finish in Photos: smart album "Keyword is cleanup_delete" → select all → ⌘⌫. The photos go to Recently Deleted for 30 days.
- Thumbnails are made from the paths in the analysis cache and kept in `~/.photo_cleanup/thumbs/`. The library's drive must be mounted.

## Setup (one time)

```bash
cd photo-cleanup
uv sync
```

- **Terminal needs access to Photos:** System Settings → Privacy & Security → *Full Disk Access* → enable Terminal (or iTerm).
- The first time keywords are set, macOS asks whether Terminal may **control Photos** → allow.
- Once in Photos: *File → New Smart Album* → "Keyword is cleanup_candidate". To compare, add a second rule "Keyword is cleanup_best" and match *any* – then each series' winner and candidates sit side by side. The album fills itself on every `scan`. (macOS 27 blocks adding photos to albums by script.)
- The first run downloads the models (SigLIP, TOPIQ, LAION, about 1–2 GB together).
- Apple's aesthetics score needs **macOS 15+**. On older systems it is skipped.

Optional, for close calls via a vision LLM:

```bash
brew install ollama && ollama serve &
ollama pull qwen3.5:9b-mlx
```

## Usage

```bash
# Start small: the newest 500 photos, report only
uv run photo_cleanup.py scan --limit 500 --no-albums

# One year, with keywords
uv run photo_cleanup.py scan --since 2025-01-01 --until 2025-12-31

# With a vision LLM as tie-breaker for close groups
uv run photo_cleanup.py scan --since 2025-01-01 --vlm
```

Or run `./trial_run.sh [N] [--library PATH]`, which installs everything and does a report-only scan of the newest N photos.

The analysis is stored in `~/.photo_cleanup/cache.sqlite`. A second run only analyzes new photos and is therefore fast. Reports (CSV with all scores) go to `~/.photo_cleanup/reports/`.

## Workflow

1. Run `scan`
2. In Photos, open the smart album (keyword **cleanup_candidate**), review it, and delete what should go with ⌘⌫
3. `uv run photo_cleanup.py learn` – the script checks which candidates you actually deleted and adapts the weighting to your taste
4. The next `scan` uses the learned weights

## Tuning

| Option | Default | Effect |
|---|---|---|
| `--sim` | 0.90 | Higher (0.94) = only group nearly identical photos. Lower (0.85) = also "same scene, different angle". Note: grouping chains (A≈B, B≈C ⇒ A, B, C in one group). At 0.90 this easily produces large groups of fairly different photos; 0.94 is a safer start. |
| `--time-gap` | 30 | Max. seconds between two photos of the same series |
| `--vlm-margin` | 0.08 | Score gap below which a group counts as close and goes to the LLM |

A good start: run with `--no-albums`, open the CSV in Numbers, spot-check a few groups, and adjust `--sim`.

## What goes into the score

| Feature | Source | Default weight |
|---|---|---|
| `topiq` | TOPIQ – technical quality (sharpness, noise, exposure) | 0.35 |
| `laion` | LAION Aesthetic Predictor – aesthetics | 0.20 |
| `apple_aes` | Apple Vision aesthetics score | 0.15 |
| `face_q` | Apple Vision face capture quality | 0.15 |
| `eyes` | Eye openness from face landmarks (worst face counts) | 0.15 |

All features are normalized to 0–1 within each group. If someone blinks, there is an extra penalty. If a model is missing, its feature is simply left out.

## Limitations

- Tested on macOS 27 with Python 3.12. On macOS 27, scripts (AppleScript or PhotoKit) cannot add photos to an album – hence keywords + a smart album.
- An honest assessment from practice: you still have to review the candidates yourself. The tool is mainly useful for finding series and getting a suggestion – whether that is faster than swiping in the Photos app depends on your library.
- Photos that live only in iCloud and have no local preview are skipped (the count is shown in the output). Workaround: in Photos → Settings → iCloud, briefly choose "Download Originals to this Mac".
- Eye detection is a heuristic and unreliable for profiles or sunglasses.
- Videos and the video part of Live Photos are not considered.
