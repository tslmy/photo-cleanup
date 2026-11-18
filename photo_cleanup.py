#!/usr/bin/env python3
"""
photo_cleanup.py – lokale Pipeline zum Aufräumen der Apple-Fotos-Mediathek (macOS).

Stufen:
  1. Index         osxphotos liest die Mediathek, Ergebnisse werden in SQLite gecacht (inkrementell)
  2. Embeddings    SigLIP-Bildvektoren für die Ähnlichkeit
  3. Gruppieren    Zeitfenster + Cosinus-Ähnlichkeit
  4. Qualität      TOPIQ (technisch), LAION (Ästhetik), Apple Vision (Ästhetik, Gesichter, Augen)
  5. Ranking       gewichtete Summe, innerhalb jeder Gruppe normalisiert
  6. Schiedsrichter lokales Vision-LLM über Ollama, nur bei knappen Gruppen (optional, --vlm)
  7. Ausgabe       Schlagwörter "bestes_foto" + "loeschkandidat" und CSV-Bericht.
                   Es wird NICHTS gelöscht – das machst du selbst in Fotos.
  8. Lernen        `learn` prüft, welche Kandidaten du wirklich gelöscht hast, und passt die Gewichte an.

Jede Stufe degradiert sauber: Fehlt ein Modell, wird das Merkmal ignoriert statt abzubrechen.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sqlite3
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np

HOME = Path.home() / ".photo_cleanup"
DB_PATH = HOME / "cache.sqlite"
WEIGHTS_PATH = HOME / "weights.json"
REPORT_DIR = HOME / "reports"

FEATURES = ["topiq", "laion", "apple_aes", "face_q", "eyes"]
DEFAULT_WEIGHTS = {"topiq": 0.35, "laion": 0.20, "apple_aes": 0.15, "face_q": 0.15, "eyes": 0.15}
EYES_CLOSED = 0.5          # eyes-Score < 0.5 ≈ Auge halb/ganz geschlossen
BLINK_PENALTY = 0.20       # Abzug, wenn jemand blinzelt
EYE_OPEN_EAR = 0.30        # Höhe/Breite eines offenen Auges (≈ 1.0 im eyes-Score)

try:  # HEIC-Unterstützung für Pillow
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


# ---------------------------------------------------------------- Hilfsfunktionen

def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def pick_device(pref: str) -> str:
    if pref != "auto":
        return pref
    try:
        import torch
        return "mps" if torch.backends.mps.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_image(path: str, max_side: int = 768):
    from PIL import Image, ImageOps
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((max_side, max_side))
    return im


def best_local_path(photo) -> str | None:
    """Lokale Vorschau bevorzugen (schnell, JPEG). Original nur, wenn keine brauchbare Vorschau da ist.
    Fotos, die nur in iCloud liegen und keine lokale Vorschau haben, werden übersprungen."""
    derivs = [d for d in (photo.path_derivatives or []) if d and Path(d).exists()]
    for d in derivs:  # osxphotos sortiert groß -> klein
        if Path(d).stat().st_size > 60_000:
            return d
    if photo.path and Path(photo.path).exists():
        return photo.path
    return derivs[0] if derivs else None


# ---------------------------------------------------------------- Datenbank

def open_db() -> sqlite3.Connection:
    HOME.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS photos(
        uuid TEXT PRIMARY KEY, ts REAL, path TEXT, filename TEXT, emb BLOB,
        topiq REAL, laion REAL, apple_aes REAL, is_utility INTEGER,
        face_q REAL, eyes REAL, faces INTEGER, error TEXT);
    CREATE TABLE IF NOT EXISTS decisions(
        run TEXT, grp INTEGER, uuid TEXT, score REAL, is_best INTEGER, feats TEXT,
        PRIMARY KEY(run, uuid));
    """)
    return con


def load_weights() -> dict:
    if WEIGHTS_PATH.exists():
        w = json.loads(WEIGHTS_PATH.read_text())
        return {f: float(w.get(f, DEFAULT_WEIGHTS[f])) for f in FEATURES}
    return dict(DEFAULT_WEIGHTS)


# ---------------------------------------------------------------- Modelle

class Embedder:
    """SigLIP-Bildembeddings (normalisiert) für die Ähnlichkeitssuche."""

    def __init__(self, device: str, model_name: str = "google/siglip-base-patch16-224"):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.proc = AutoImageProcessor.from_pretrained(model_name)

    def __call__(self, images) -> np.ndarray:
        with self.torch.no_grad():
            inp = self.proc(images=images, return_tensors="pt").to(self.device)
            f = self.model.get_image_features(**inp)
            if not isinstance(f, self.torch.Tensor):  # transformers 5 liefert ein Output-Objekt
                f = f.pooler_output
            f = f / f.norm(dim=-1, keepdim=True)
        return f.float().cpu().numpy().astype(np.float32)


class IQA:
    """Bildqualität über pyiqa: TOPIQ (technisch) + LAION (Ästhetik). Fällt bei MPS-Problemen auf CPU zurück."""

    def __init__(self, device: str):
        self.metrics = {}
        self.names = {"topiq": "topiq_nr", "laion": "laion_aes"}
        try:
            import pyiqa
            import torchvision.transforms.functional as TF
            self.TF = TF
            self.pyiqa = pyiqa
        except ImportError:
            log("pyiqa nicht installiert – TOPIQ/LAION werden übersprungen")
            return
        for key, name in self.names.items():
            for dev in dict.fromkeys([device, "cpu"]):
                try:
                    self.metrics[key] = pyiqa.create_metric(name, device=dev)
                    break
                except Exception as e:  # noqa: BLE001
                    log(f"{name} auf {dev} nicht verfügbar: {e}")

    def __call__(self, im) -> dict:
        out = {"topiq": None, "laion": None}
        if not self.metrics:
            return out
        t = self.TF.to_tensor(im).unsqueeze(0)
        for key, m in list(self.metrics.items()):
            try:
                out[key] = float(m(t.to(m.device)).item())
            except Exception as e:  # noqa: BLE001
                if str(m.device) != "cpu":  # z. B. fehlende MPS-Ops: auf CPU weiter
                    log(f"{key} auf {m.device} fehlgeschlagen – weiter auf CPU")
                    try:
                        m = self.metrics[key] = self.pyiqa.create_metric(self.names[key], device="cpu")
                        out[key] = float(m(t).item())
                        continue
                    except Exception as e2:  # noqa: BLE001
                        e = e2
                log(f"{key} fehlgeschlagen ({e}) – Merkmal wird deaktiviert")
                self.metrics.pop(key)
        return out


def _points(region):
    n = int(region.pointCount())
    pts = region.normalizedPoints()
    res = []
    for i in range(n):
        p = pts[i]
        res.append((float(p.x), float(p.y)) if hasattr(p, "x") else (float(p[0]), float(p[1])))
    return res


def _eye_ratio(points) -> float | None:
    if len(points) < 4:
        return None
    xs, ys = zip(*points)
    w = max(xs) - min(xs)
    return (max(ys) - min(ys)) / w if w > 0 else None


def apple_vision_scores(path: str) -> dict:
    """Apple Vision über PyObjC: Ästhetik-Score (macOS 15+), Gesichtsqualität, Augenöffnung."""
    out = {"apple_aes": None, "is_utility": None, "face_q": None, "eyes": None, "faces": 0}
    try:
        import Vision
        from Foundation import NSURL
    except ImportError:
        return out
    try:
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(path), None)  # pyobjc 13: {} -> "key does not exist"
        aes_req = None
        reqs = []
        if hasattr(Vision, "VNCalculateImageAestheticsScoresRequest"):
            aes_req = Vision.VNCalculateImageAestheticsScoresRequest.alloc().init()
            reqs.append(aes_req)
        fq_req = Vision.VNDetectFaceCaptureQualityRequest.alloc().init()
        lm_req = Vision.VNDetectFaceLandmarksRequest.alloc().init()
        reqs += [fq_req, lm_req]
        ok, _err = handler.performRequests_error_(reqs, None)
        if not ok:
            return out

        if aes_req is not None and aes_req.results():
            obs = aes_req.results()[0]
            out["apple_aes"] = float(obs.overallScore())
            out["is_utility"] = bool(obs.isUtility())

        qs = [float(o.faceCaptureQuality()) for o in (fq_req.results() or [])
              if o.faceCaptureQuality() is not None]
        out["faces"] = len(qs)
        if qs:
            out["face_q"] = float(np.mean(qs))

        # Augen: schlechtestes Gesicht zählt (einer blinzelt = Foto schlechter)
        worst = None
        for o in (lm_req.results() or []):
            lm = o.landmarks()
            if lm is None or lm.leftEye() is None or lm.rightEye() is None:
                continue
            bb = o.boundingBox()
            aspect = bb.size.height / bb.size.width if bb.size.width else 1.0
            ratios = [_eye_ratio(_points(e)) for e in (lm.leftEye(), lm.rightEye())]
            ratios = [r * aspect for r in ratios if r is not None]
            if ratios:
                score = min(1.0, min(ratios) / EYE_OPEN_EAR)
                worst = score if worst is None else min(worst, score)
        out["eyes"] = worst
    except Exception as e:  # noqa: BLE001
        log(f"Vision-Fehler bei {Path(path).name}: {e}")
    return out


# ---------------------------------------------------------------- Stufe 1–4: Index + Analyse

LIBRARY: str | None = None  # set from --library in main()


def resolve_library() -> str:
    """The library to read. Explicit --library wins; otherwise the one Photos opened last."""
    from osxphotos.utils import get_last_library_path, get_system_library_path
    lib = LIBRARY or get_last_library_path() or get_system_library_path()
    if not lib or not Path(lib).exists():
        sys.exit(f"Photos library not found: {lib!r} (is the drive mounted? use --library PATH)")
    return str(Path(lib).resolve())


def open_photos_db():
    import osxphotos
    lib = resolve_library()
    log(f"Library: {lib}")
    return osxphotos.PhotosDB(dbfile=lib)


def check_photos_app_library() -> bool:
    """AppleScript writes to whatever library the Photos app has open, not to --library.
    Refuse to tag unless they are the same library."""
    from osxphotos.utils import get_last_library_path
    target = resolve_library()
    current = get_last_library_path()
    if not current or Path(current).resolve() != Path(target):
        log(f"NOT tagging: Photos app last opened {current!r}, but the scan used {target!r}. "
            "Open the right library in Photos (hold ⌥ while launching Photos) and re-run.")
        return False
    return True


def library_photos(args):
    log("Lese Fotos-Mediathek …")
    db = open_photos_db()
    photos = [p for p in db.photos(images=True, movies=False)
              if not p.intrash and not p.hidden and p.date is not None]
    if args.since:
        since = date.fromisoformat(args.since)
        photos = [p for p in photos if p.date.date() >= since]
    if args.until:
        until = date.fromisoformat(args.until)
        photos = [p for p in photos if p.date.date() <= until]
    photos.sort(key=lambda p: p.date)
    if args.limit:
        photos = photos[-args.limit:]
    log(f"{len(photos)} Fotos im Auswahlbereich")
    return photos


def analyze(con, photos, args) -> None:
    todo, skipped = [], 0
    for p in photos:
        row = con.execute("SELECT emb FROM photos WHERE uuid=?", (p.uuid,)).fetchone()
        if row and row[0] is not None:
            continue
        path = best_local_path(p)
        if not path:
            skipped += 1
            continue
        todo.append((p, path))
    log(f"{len(todo)} neue Fotos zu analysieren, {skipped} ohne lokale Datei übersprungen")
    if not todo:
        return

    device = pick_device(args.device)
    log(f"Lade Modelle auf {device} …")
    embedder = Embedder(device)
    iqa = IQA(device)

    t0, done = time.time(), 0
    for chunk in batched(todo, 16):
        imgs, ok = [], []
        for p, path in chunk:
            try:
                imgs.append(load_image(path))
                ok.append((p, path))
            except Exception as e:  # noqa: BLE001
                con.execute("INSERT OR REPLACE INTO photos(uuid, ts, path, filename, error) VALUES(?,?,?,?,?)",
                            (p.uuid, p.date.timestamp(), path, p.original_filename, str(e)))
        if imgs:
            embs = embedder(imgs)
            for (p, path), im, e in zip(ok, imgs, embs):
                q = iqa(im)
                v = apple_vision_scores(path)
                con.execute(
                    "INSERT OR REPLACE INTO photos VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (p.uuid, p.date.timestamp(), path, p.original_filename, e.tobytes(),
                     q["topiq"], q["laion"], v["apple_aes"],
                     None if v["is_utility"] is None else int(v["is_utility"]),
                     v["face_q"], v["eyes"], v["faces"], None))
        con.commit()
        done += len(chunk)
        rate = done / max(time.time() - t0, 1e-6)
        log(f"{done}/{len(todo)} analysiert ({rate:.1f} Fotos/s, Rest ≈ {(len(todo) - done) / rate / 60:.0f} min)")


# ---------------------------------------------------------------- Stufe 3: Gruppieren

@dataclass
class Item:
    uuid: str
    ts: float
    path: str
    filename: str
    emb: np.ndarray
    raw: dict
    is_utility: bool = False
    norm: dict = field(default_factory=dict)
    score: float = 0.0


def load_items(con, uuids: set[str]) -> list[Item]:
    items = []
    for r in con.execute("SELECT uuid, ts, path, filename, emb, topiq, laion, apple_aes, is_utility, "
                         "face_q, eyes FROM photos WHERE emb IS NOT NULL"):
        if r[0] not in uuids:
            continue
        items.append(Item(uuid=r[0], ts=r[1], path=r[2], filename=r[3] or "",
                          emb=np.frombuffer(r[4], dtype=np.float32),
                          raw=dict(zip(FEATURES, (r[5], r[6], r[7], r[9], r[10]))),
                          is_utility=bool(r[8])))
    items.sort(key=lambda i: i.ts)
    return items


def group_items(items: list[Item], time_gap: float, sim_thr: float, max_window: int = 80) -> list[list[Item]]:
    """Fotos, die zeitlich nah beieinander liegen, bilden ein Fenster.
    Innerhalb eines Fensters werden Paare mit Cosinus-Ähnlichkeit >= sim_thr verbunden (Union-Find)."""
    groups: list[list[Item]] = []

    def flush(win: list[Item]):
        if len(win) < 2:
            return
        E = np.stack([w.emb for w in win])
        S = E @ E.T
        parent = list(range(len(win)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(win)):
            for j in range(i + 1, len(win)):
                if S[i, j] >= sim_thr:
                    parent[find(i)] = find(j)
        comps: dict[int, list[Item]] = {}
        for i, w in enumerate(win):
            comps.setdefault(find(i), []).append(w)
        groups.extend(c for c in comps.values() if len(c) >= 2)

    if not items:
        return groups
    window = [items[0]]
    for it in items[1:]:
        if it.ts - window[-1].ts <= time_gap and len(window) < max_window:
            window.append(it)
        else:
            flush(window)
            window = [it]
    flush(window)
    return groups


# ---------------------------------------------------------------- Stufe 5: Ranking

def rank_group(group: list[Item], weights: dict) -> list[Item]:
    """Min-Max-Normalisierung je Merkmal innerhalb der Gruppe, dann gewichtete Summe.
    Merkmale, die in der ganzen Gruppe fehlen (z. B. keine Gesichter), zählen nicht mit."""
    used = {}
    for f in FEATURES:
        vals = [g.raw[f] for g in group if g.raw[f] is not None]
        if not vals or weights.get(f, 0) <= 0:
            for g in group:
                g.norm[f] = 0.5
            continue
        lo, hi, med = min(vals), max(vals), float(np.median(vals))
        for g in group:
            v = g.raw[f] if g.raw[f] is not None else med
            g.norm[f] = 0.5 if hi - lo < 1e-6 else (v - lo) / (hi - lo)
        used[f] = weights[f]
    total = sum(used.values()) or 1.0
    for g in group:
        g.score = sum(w * g.norm[f] for f, w in used.items()) / total
        if g.raw["eyes"] is not None and g.raw["eyes"] < EYES_CLOSED:
            g.score -= BLINK_PENALTY
    return sorted(group, key=lambda g: g.score, reverse=True)


# ---------------------------------------------------------------- Stufe 6: Vision-LLM

def _encode(path: str, max_side: int = 768) -> str:
    buf = io.BytesIO()
    load_image(path, max_side).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def ollama_json(prompt: str, images: list[str], model: str, host: str) -> dict:
    """One Ollama chat call that must answer with a JSON object. think=False keeps
    reasoning models (qwen3.5) from spending minutes thinking before they answer.
    No "format": "json" — Ollama's MLX runner rejects it (HTTP 501), so the object is cut out of the text."""
    body = {"model": model, "stream": False, "think": False,
            "options": {"temperature": 0},
            "messages": [{"role": "user", "content": prompt, "images": images}]}
    req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        text = json.load(r)["message"]["content"]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"no JSON in VLM answer: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def vlm_pick(cands: list[Item], model: str, host: str) -> tuple[int, str] | None:
    imgs = [_encode(c.path) for c in cands]
    prompt = (f"You see {len(cands)} very similar photos, numbered 1 to {len(cands)} in the given order. "
              "Which one should be kept? Consider sharpness, open eyes, facial expression, framing and exposure. "
              'Answer only as JSON: {"best": <number>, "reason": "<one short sentence>"}')
    try:
        data = ollama_json(prompt, imgs, model, host)
        idx = int(data["best"]) - 1
        if 0 <= idx < len(cands):
            return idx, str(data.get("reason", ""))
    except Exception as e:  # noqa: BLE001
        log(f"Vision-LLM fehlgeschlagen: {e}")
    return None


# ---------------------------------------------------------------- Stufe 7: Ausgabe

CAND_KEYWORD = "loeschkandidat"
BEST_KEYWORD = "bestes_foto"

_KEYWORD_SCRIPT = """
on run argv
  set mode to item 1 of argv
  set kwd to item 2 of argv
  set n to 0
  tell application "Photos"
    repeat with i from 3 to count of argv
      set m to media item id ((item i of argv) & "/L0/001")
      set kw to keywords of m
      if kw is missing value then set kw to {}
      if mode is "add" then
        if kw does not contain kwd then set keywords of m to kw & {kwd}
      else
        set kept to {}
        repeat with k in kw
          if (contents of k) is not kwd then set end of kept to (contents of k)
        end repeat
        set keywords of m to kept
      end if
      set n to n + 1
    end repeat
  end tell
  return n
end run
"""


def tag_photos(keyword: str, uuids: list[str], scope: set[str]) -> None:
    """Gibt genau diesen Fotos das Schlagwort; ein intelligentes Album in Fotos zeigt sie an.
    Patched: the keyword is only removed from photos inside `scope` (this run's date range), so scanning
    one year at a time no longer wipes earlier years' tags. IDs are sent in batches to stay under ARG_MAX.
    Fotos per Skript in Alben legen geht unter macOS 27 nicht (PhotoKit und AppleScript werden still abgelehnt),
    Schlagwörter setzen schon."""
    import subprocess

    def run(mode, ids):
        n = 0
        for chunk in batched(ids, 500):
            r = subprocess.run(["osascript", "-", mode, keyword, *chunk], input=_KEYWORD_SCRIPT,
                               text=True, capture_output=True)
            if r.returncode:
                log(f"Schlagwort „{keyword}“ ({mode}) fehlgeschlagen: {r.stderr.strip()}")
                continue
            n += int(r.stdout.strip() or 0)
        return n

    old = {p.uuid for p in open_photos_db().photos(keywords=[keyword])}
    removed = run("remove", sorted((old & scope) - set(uuids)))
    added = run("add", sorted(set(uuids) - old))
    log(f"Schlagwort „{keyword}“: {added} neu, {len(set(uuids) & old)} schon da, {removed} entfernt")


def cmd_scan(args) -> None:
    con = open_db()
    photos = library_photos(args)
    analyze(con, photos, args)

    items = load_items(con, {p.uuid for p in photos})
    groups = group_items(items, args.time_gap, args.sim)
    log(f"{len(groups)} Gruppen ähnlicher Fotos gefunden "
        f"({sum(len(g) for g in groups)} Fotos)")

    weights = load_weights()
    run = datetime.now().strftime("%Y-%m-%d %H-%M")
    best_uuids, cand_uuids, report = [], [], []
    vlm_used = 0

    for gi, group in enumerate(groups):
        ranked = rank_group(group, weights)
        reason = ""
        close = len(ranked) > 1 and ranked[0].score - ranked[1].score < args.vlm_margin
        if args.vlm and close:
            top = ranked[:4]
            pick = vlm_pick(top, args.vlm_model, args.ollama)
            if pick:
                idx, reason = pick
                vlm_used += 1
                ranked.insert(0, ranked.pop(idx))
        best = ranked[0]
        best_uuids.append(best.uuid)
        for rank, it in enumerate(ranked):
            is_best = rank == 0
            if not is_best:
                cand_uuids.append(it.uuid)
            con.execute("INSERT OR REPLACE INTO decisions VALUES(?,?,?,?,?,?)",
                        (run, gi, it.uuid, it.score, int(is_best), json.dumps(it.norm)))
            report.append({
                "gruppe": gi + 1, "rang": rank + 1, "behalten": "ja" if is_best else "",
                "datei": it.filename, "datum": datetime.fromtimestamp(it.ts).isoformat(timespec="seconds"),
                "score": round(it.score, 3),
                **{f: (None if it.raw[f] is None else round(it.raw[f], 3)) for f in FEATURES},
                "knapp": "ja" if close else "", "llm_begruendung": reason if is_best else "",
                "uuid": it.uuid,
            })
    con.commit()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / f"bericht {run}.csv"
    if report:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(report[0].keys()), delimiter=";")
            w.writeheader()
            w.writerows(report)
        log(f"Bericht: {csv_path}")

    if vlm_used:
        log(f"Vision-LLM hat in {vlm_used} knappen Gruppen entschieden")
    if args.no_albums:
        log("--no-albums: keine Schlagwörter gesetzt")
    elif groups and check_photos_app_library():
        scope = {p.uuid for p in photos}
        tag_photos(CAND_KEYWORD, cand_uuids, scope)
        tag_photos(BEST_KEYWORD, best_uuids, scope)
        log(f"In Fotos: Ablage → Neues intelligentes Album → Schlagwort ist „{CAND_KEYWORD}“ "
            f"(zum Vergleichen: „Beliebige“ Regel + Schlagwort ist „{BEST_KEYWORD}“)")
    log(f"Fertig: {len(best_uuids)} beste Fotos, {len(cand_uuids)} Löschkandidaten. Gelöscht wurde nichts.")


# ---------------------------------------------------------------- Added: standalone low-quality photos

LOWQ_KEYWORD = "cleanup_lowq"

_LOWQ_PROMPT = """This photo was taken in {year}{camera}. An automatic quality check flagged it ({flags}).
Decide whether it is worth keeping in a family photo library.
Answer "delete" ONLY if it has no value: accidental shot (pocket, floor, ceiling, lens cap), nearly black or white,
so blurred that nothing is recognisable, or a throwaway screenshot / receipt / document photo.
Answer "keep" if it shows people, pets, a place or event someone might want to remember, even if it is
blurry, noisy, low-resolution or badly exposed. Old cameras were bad; judge by content, not technical quality.
Answer "unsure" if you cannot tell.
Answer only as JSON: {{"verdict": "delete" | "keep" | "unsure", "reason": "<one short sentence>"}}"""


def cmd_lowq(args) -> None:
    """Flag photos that are low quality on their own (not just the worse shot of a series).
    Uses the scores `scan` already caches; optionally asks the local VLM to confirm each candidate."""
    con = open_db()
    photos = library_photos(args)
    analyze(con, photos, args)
    meta = {p.uuid: p for p in photos}

    rows = con.execute("SELECT uuid, ts, path, filename, topiq, laion, apple_aes, is_utility "
                       "FROM photos WHERE emb IS NOT NULL").fetchall()
    cands = []
    for uuid, ts, path, fn, topiq, laion, aes, util in rows:
        if uuid not in meta:
            continue
        flags = []
        if util:
            flags.append("utility")  # Apple Vision: screenshot, document, receipt …
        if topiq is not None and topiq < args.topiq_max:
            flags.append("low_technical")  # blur, noise, exposure
        if aes is not None and aes < args.aes_max:
            flags.append("low_aesthetic")
        if flags:
            cands.append(dict(uuid=uuid, ts=ts, path=path, filename=fn, flags=flags,
                              topiq=topiq, laion=laion, apple_aes=aes))
    cands.sort(key=lambda c: (c["topiq"] if c["topiq"] is not None else 1.0))
    log(f"{len(cands)} of {len(meta)} photos flagged "
        f"(topiq < {args.topiq_max}, apple_aes < {args.aes_max}, or utility)")

    t0 = time.time()
    for i, c in enumerate(cands, 1):
        c["verdict"], c["reason"] = ("candidate", "")
        if not args.vlm:
            continue
        p = meta[c["uuid"]]
        cam = getattr(p.exif_info, "camera_model", None) if p.exif_info else None
        prompt = _LOWQ_PROMPT.format(year=datetime.fromtimestamp(c["ts"]).year,
                                     camera=f" with a {cam}" if cam else "", flags=", ".join(c["flags"]))
        try:
            d = ollama_json(prompt, [_encode(c["path"])], args.vlm_model, args.ollama)
            c["verdict"] = str(d.get("verdict", "unsure")).lower()
            c["reason"] = str(d.get("reason", ""))
        except Exception as e:  # noqa: BLE001
            c["verdict"], c["reason"] = "error", str(e)
        if i % 10 == 0 or i == len(cands):
            rate = i / max(time.time() - t0, 1e-6)
            log(f"VLM {i}/{len(cands)} ({rate:.2f} photos/s)")

    run = datetime.now().strftime("%Y-%m-%d %H-%M")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / f"lowq {run}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        cols = ["verdict", "reason", "flags", "date", "filename", "topiq", "laion", "apple_aes", "uuid", "path"]
        w = csv.DictWriter(fh, fieldnames=cols, delimiter=";")
        w.writeheader()
        for c in cands:
            w.writerow({**{k: c[k] for k in cols if k in c}, "flags": ",".join(c["flags"]),
                        "date": datetime.fromtimestamp(c["ts"]).isoformat(timespec="seconds"),
                        **{k: None if c[k] is None else round(c[k], 3) for k in ("topiq", "laion", "apple_aes")}})
    log(f"Report: {csv_path}")
    counts: dict[str, int] = {}
    for c in cands:
        counts[c["verdict"]] = counts.get(c["verdict"], 0) + 1
    log("Verdicts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    to_tag = [c["uuid"] for c in cands if c["verdict"] in ("delete", "candidate")]
    if args.no_albums:
        log("--no-albums: no keywords set")
    elif check_photos_app_library():
        tag_photos(LOWQ_KEYWORD, to_tag, set(meta))
        log(f"In Photos: File → New Smart Album → Keyword is “{LOWQ_KEYWORD}”")
    log("Nothing was deleted.")


# ---------------------------------------------------------------- Stufe 8: Lernen

def cmd_learn(args) -> None:
    from sklearn.linear_model import LogisticRegression

    con = open_db()
    db = open_photos_db()
    rows = con.execute("SELECT run, grp, uuid, feats FROM decisions").fetchall()

    by_group: dict[tuple, list] = {}
    for run, grp, uuid, feats in rows:
        p = db.get_photo(uuid)
        deleted = p is None or p.intrash
        by_group.setdefault((run, grp), []).append((json.loads(feats), deleted))

    X, y = [], []
    for members in by_group.values():
        # nur Gruppen, die du sichtbar durchgesehen hast (mind. ein Foto gelöscht, mind. eins behalten)
        flags = [d for _, d in members]
        if any(flags) and not all(flags):
            for feats, deleted in members:
                X.append([feats.get(f, 0.5) for f in FEATURES])
                y.append(0 if deleted else 1)

    log(f"{len(y)} Entscheidungen aus durchgesehenen Gruppen")
    if len(y) < 40 or len(set(y)) < 2:
        log("Noch zu wenig Daten – erst ein paar Gruppen in Fotos durchsehen und Kandidaten löschen.")
        return

    clf = LogisticRegression(max_iter=1000).fit(np.array(X), np.array(y))
    coefs = np.clip(clf.coef_[0], 0, None)
    if coefs.sum() == 0:
        log("Kein klares Muster gefunden – Gewichte bleiben unverändert.")
        return
    learned = coefs / coefs.sum()
    # Mit Standardgewichten mischen, damit wenige Entscheidungen nicht überanpassen
    alpha = len(y) / (len(y) + 200)
    new = {f: float(alpha * learned[i] + (1 - alpha) * DEFAULT_WEIGHTS[f]) for i, f in enumerate(FEATURES)}
    HOME.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH.write_text(json.dumps(new, indent=2))
    log("Neue Gewichte: " + ", ".join(f"{f}={w:.2f}" for f, w in new.items()))


# ---------------------------------------------------------------- CLI

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Ähnliche Fotos finden und das beste auswählen (lokal).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    lib = argparse.ArgumentParser(add_help=False)
    lib.add_argument("--library", help="path to the .photoslibrary (default: the one Photos opened last)")

    common = argparse.ArgumentParser(add_help=False, parents=[lib])
    common.add_argument("--since", help="nur Fotos ab Datum (YYYY-MM-DD)")
    common.add_argument("--until", help="nur Fotos bis Datum (YYYY-MM-DD)")
    common.add_argument("--limit", type=int, help="nur die neuesten N Fotos")
    common.add_argument("--device", default="auto", help="auto | mps | cpu")
    common.add_argument("--vlm-model", default="qwen3.5:9b-mlx")
    common.add_argument("--ollama", default="http://localhost:11434")
    common.add_argument("--no-albums", action="store_true", help="nur CSV-Bericht, keine Schlagwörter in Fotos")

    s = sub.add_parser("scan", parents=[common], help="similar photos: pick the best, tag the rest")
    s.add_argument("--time-gap", type=float, default=30, help="max. Sekunden zwischen Fotos einer Serie (30)")
    s.add_argument("--sim", type=float, default=0.90, help="Ähnlichkeitsschwelle 0–1 (0.90)")
    s.add_argument("--vlm", action="store_true", help="knappe Gruppen von lokalem Vision-LLM entscheiden lassen")
    s.add_argument("--vlm-margin", type=float, default=0.08, help="Score-Abstand, ab dem eine Gruppe als knapp gilt")
    s.set_defaults(func=cmd_scan)

    q = sub.add_parser("lowq", parents=[common], help="standalone low-quality photos")
    q.add_argument("--topiq-max", type=float, default=0.30, help="flag if TOPIQ technical score is below (0.30)")
    q.add_argument("--aes-max", type=float, default=-0.30, help="flag if Apple aesthetic score is below (-0.30)")
    q.add_argument("--vlm", action="store_true", help="ask the local VLM to confirm each flagged photo")
    q.set_defaults(func=cmd_lowq)

    l = sub.add_parser("learn", parents=[lib], help="Gewichte aus deinen Lösch-Entscheidungen lernen")
    l.set_defaults(func=cmd_learn)

    args = ap.parse_args(argv)
    if sys.platform != "darwin":
        sys.exit("Dieses Skript läuft nur auf macOS (Fotos-Mediathek).")
    global LIBRARY
    LIBRARY = args.library
    args.func(args)


if __name__ == "__main__":
    main()
