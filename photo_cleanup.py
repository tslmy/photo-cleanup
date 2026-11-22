#!/usr/bin/env python3
"""
photo_cleanup.py – local pipeline for tidying up an Apple Photos library (macOS).

Stages:
  1. Index         osxphotos reads the library; results are cached in SQLite (incremental)
  2. Embeddings    SigLIP image vectors for similarity
  3. Grouping      time window + cosine similarity
  4. Quality       TOPIQ (technical), LAION (aesthetics), Apple Vision (aesthetics, faces, eyes)
  5. Ranking       weighted sum, normalized within each group
  6. Tie-breaker   local vision LLM via Ollama, only for close groups (optional, --vlm)
  7. Output        keywords "cleanup_best" + "cleanup_candidate" and a CSV report.
                   NOTHING is deleted – you do that yourself in Photos.
  8. Learning      `learn` checks which candidates you actually deleted and adjusts the weights.

Every stage degrades gracefully: if a model is missing, its feature is ignored instead of aborting.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import sqlite3
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np

HOME = Path(os.environ.get("PHOTO_CLEANUP_HOME") or Path.home() / ".photo_cleanup")
DB_PATH = HOME / "cache.sqlite"
WEIGHTS_PATH = HOME / "weights.json"
REPORT_DIR = HOME / "reports"

FEATURES = ["topiq", "laion", "apple_aes", "face_q", "eyes"]
DEFAULT_WEIGHTS = {"topiq": 0.35, "laion": 0.20, "apple_aes": 0.15, "face_q": 0.15, "eyes": 0.15}
EYES_CLOSED = 0.5          # eyes score < 0.5 ≈ eye half or fully closed
BLINK_PENALTY = 0.20       # penalty when someone blinks
EYE_OPEN_EAR = 0.30        # height/width of an open eye (≈ 1.0 in the eyes score)

try:  # HEIC support for Pillow
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


# ---------------------------------------------------------------- Helpers

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
    """Prefer a local preview (fast, JPEG). Use the original only when no usable preview exists.
    Photos that live only in iCloud and have no local preview are skipped."""
    derivs = [d for d in (photo.path_derivatives or []) if d and Path(d).exists()]
    for d in derivs:  # osxphotos sorts large -> small
        if Path(d).stat().st_size > 60_000:
            return d
    if photo.path and Path(photo.path).exists():
        return photo.path
    return derivs[0] if derivs else None


# ---------------------------------------------------------------- Database

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


# ---------------------------------------------------------------- Models

class Embedder:
    """SigLIP image embeddings (normalized) for similarity search."""

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
            if not isinstance(f, self.torch.Tensor):  # transformers 5 returns an output object
                f = f.pooler_output
            f = f / f.norm(dim=-1, keepdim=True)
        return f.float().cpu().numpy().astype(np.float32)


class IQA:
    """Image quality via pyiqa: TOPIQ (technical) + LAION (aesthetics). Falls back to CPU on MPS problems."""

    def __init__(self, device: str):
        self.metrics = {}
        self.names = {"topiq": "topiq_nr", "laion": "laion_aes"}
        try:
            import pyiqa
            import torchvision.transforms.functional as TF
            self.TF = TF
            self.pyiqa = pyiqa
        except ImportError:
            log("pyiqa not installed – skipping TOPIQ/LAION")
            return
        for key, name in self.names.items():
            for dev in dict.fromkeys([device, "cpu"]):
                try:
                    self.metrics[key] = pyiqa.create_metric(name, device=dev)
                    break
                except Exception as e:  # noqa: BLE001
                    log(f"{name} not available on {dev}: {e}")

    def __call__(self, im) -> dict:
        out = {"topiq": None, "laion": None}
        if not self.metrics:
            return out
        t = self.TF.to_tensor(im).unsqueeze(0)
        for key, m in list(self.metrics.items()):
            try:
                out[key] = float(m(t.to(m.device)).item())
            except Exception:  # noqa: BLE001
                if str(m.device) != "cpu":  # e.g. missing MPS ops: continue on CPU
                    log(f"{key} failed on {m.device} – continuing on CPU")
                    try:
                        m = self.metrics[key] = self.pyiqa.create_metric(self.names[key], device="cpu")
                        out[key] = float(m(t).item())
                        continue
                    except Exception as e2:  # noqa: BLE001
                        e = e2
                log(f"{key} failed ({e}) – disabling feature")
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
    """Apple Vision via PyObjC: aesthetics score (macOS 15+), face capture quality, eye openness."""
    out = {"apple_aes": None, "is_utility": None, "face_q": None, "eyes": None, "faces": 0}
    try:
        import Vision
        from Foundation import NSURL
    except ImportError:
        return out
    try:
        # pyobjc 13: passing {} as options -> "key does not exist", so pass None
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(path), None)
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

        # Eyes: the worst face counts (one person blinking = worse photo)
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
        log(f"Vision error on {Path(path).name}: {e}")
    return out


# ---------------------------------------------------------------- Stages 1–4: index + analysis

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
    log("Reading Photos library …")
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
    log(f"{len(photos)} photos in the selected range")
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
    log(f"{len(todo)} new photos to analyze, {skipped} skipped without a local file")
    if not todo:
        return

    device = pick_device(args.device)
    log(f"Loading models on {device} …")
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
        log(f"{done}/{len(todo)} analyzed ({rate:.1f} photos/s, ≈ {(len(todo) - done) / rate / 60:.0f} min left)")


# ---------------------------------------------------------------- Stage 3: grouping

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
    """Photos taken close together in time form a window.
    Within a window, pairs with cosine similarity >= sim_thr are linked (union-find)."""
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


# ---------------------------------------------------------------- Stage 5: ranking

def rank_group(group: list[Item], weights: dict) -> list[Item]:
    """Min-max normalization per feature within the group, then a weighted sum.
    Features missing across the whole group (e.g. no faces) are not counted."""
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


# ---------------------------------------------------------------- Stage 6: vision LLM

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
        log(f"Vision LLM failed: {e}")
    return None


# ---------------------------------------------------------------- Stage 7: output

CAND_KEYWORD = "cleanup_candidate"
BEST_KEYWORD = "cleanup_best"

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


def tag_photos(keyword: str, uuids: list[str], scope: set[str]) -> dict:
    """Give exactly these photos the keyword; a smart album in Photos shows them.
    The keyword is only removed from photos inside `scope` (this run's date range), so scanning
    one year at a time doesn't wipe earlier years' tags. IDs are sent in batches to stay under ARG_MAX.
    Adding photos to albums by script doesn't work on macOS 27 (PhotoKit and AppleScript are silently
    rejected), but setting keywords does."""
    import subprocess

    def run(mode, ids):
        n = 0
        for chunk in batched(ids, 500):
            r = subprocess.run(["osascript", "-", mode, keyword, *chunk], input=_KEYWORD_SCRIPT,
                               text=True, capture_output=True)
            if r.returncode:
                log(f"Keyword “{keyword}” ({mode}) failed: {r.stderr.strip()}")
                continue
            n += int(r.stdout.strip() or 0)
        return n

    old = {p.uuid for p in open_photos_db().photos(keywords=[keyword])}
    removed = run("remove", sorted((old & scope) - set(uuids)))
    added = run("add", sorted(set(uuids) - old))
    present = len(set(uuids) & old)
    log(f"Keyword “{keyword}”: {added} added, {present} already present, {removed} removed")
    return {"added": added, "present": present, "removed": removed}


def cmd_scan(args) -> None:
    con = open_db()
    photos = library_photos(args)
    analyze(con, photos, args)

    items = load_items(con, {p.uuid for p in photos})
    groups = group_items(items, args.time_gap, args.sim)
    log(f"Found {len(groups)} groups of similar photos "
        f"({sum(len(g) for g in groups)} photos)")

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
                "group": gi + 1, "rank": rank + 1, "keep": "yes" if is_best else "",
                "filename": it.filename, "date": datetime.fromtimestamp(it.ts).isoformat(timespec="seconds"),
                "score": round(it.score, 3),
                **{f: (None if it.raw[f] is None else round(it.raw[f], 3)) for f in FEATURES},
                "close": "yes" if close else "", "vlm_reason": reason if is_best else "",
                "uuid": it.uuid,
            })
    con.commit()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / f"report {run}.csv"
    if report:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(report[0].keys()), delimiter=";")
            w.writeheader()
            w.writerows(report)
        log(f"Report: {csv_path}")

    if vlm_used:
        log(f"Vision LLM decided {vlm_used} close groups")
    if args.no_albums:
        log("--no-albums: no keywords set")
    elif groups and check_photos_app_library():
        scope = {p.uuid for p in photos}
        tag_photos(CAND_KEYWORD, cand_uuids, scope)
        tag_photos(BEST_KEYWORD, best_uuids, scope)
        log(f"In Photos: File → New Smart Album → Keyword is “{CAND_KEYWORD}” "
            f"(to compare: match “any” rule + Keyword is “{BEST_KEYWORD}”)")
    log(f"Done: {len(best_uuids)} best photos, {len(cand_uuids)} deletion candidates. Nothing was deleted.")


# ---------------------------------------------------------------- Standalone low-quality photos

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


# ---------------------------------------------------------------- Stage 8: learning

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
        # only groups you visibly reviewed (at least one photo deleted, at least one kept)
        flags = [d for _, d in members]
        if any(flags) and not all(flags):
            for feats, deleted in members:
                X.append([feats.get(f, 0.5) for f in FEATURES])
                y.append(0 if deleted else 1)

    log(f"{len(y)} decisions from reviewed groups")
    if len(y) < 40 or len(set(y)) < 2:
        log("Not enough data yet – review a few groups in Photos and delete candidates first.")
        return

    clf = LogisticRegression(max_iter=1000).fit(np.array(X), np.array(y))
    coefs = np.clip(clf.coef_[0], 0, None)
    if coefs.sum() == 0:
        log("No clear pattern found – weights unchanged.")
        return
    learned = coefs / coefs.sum()
    # Blend with the default weights so a few decisions don't overfit
    alpha = len(y) / (len(y) + 200)
    new = {f: float(alpha * learned[i] + (1 - alpha) * DEFAULT_WEIGHTS[f]) for i, f in enumerate(FEATURES)}
    HOME.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH.write_text(json.dumps(new, indent=2))
    log("New weights: " + ", ".join(f"{f}={w:.2f}" for f, w in new.items()))


# ---------------------------------------------------------------- CLI

def cmd_serve(args) -> None:
    import review_server
    # Hand over this module object: when run as a script it is __main__, and a plain
    # `import photo_cleanup` inside review_server would create a second copy with its own LIBRARY.
    review_server.serve(sys.modules[__name__], args)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Find similar photos and pick the best one (local).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    lib = argparse.ArgumentParser(add_help=False)
    lib.add_argument("--library", help="path to the .photoslibrary (default: the one Photos opened last)")

    common = argparse.ArgumentParser(add_help=False, parents=[lib])
    common.add_argument("--since", help="only photos on or after this date (YYYY-MM-DD)")
    common.add_argument("--until", help="only photos on or before this date (YYYY-MM-DD)")
    common.add_argument("--limit", type=int, help="only the newest N photos")
    common.add_argument("--device", default="auto", help="auto | mps | cpu")
    common.add_argument("--vlm-model", default="qwen3.5:9b-mlx")
    common.add_argument("--ollama", default="http://localhost:11434")
    common.add_argument("--no-albums", action="store_true", help="CSV report only, no keywords in Photos")

    s = sub.add_parser("scan", parents=[common], help="similar photos: pick the best, tag the rest")
    s.add_argument("--time-gap", type=float, default=30, help="max. seconds between photos of a series (30)")
    s.add_argument("--sim", type=float, default=0.90, help="similarity threshold 0–1 (0.90)")
    s.add_argument("--vlm", action="store_true", help="let a local vision LLM decide close groups")
    s.add_argument("--vlm-margin", type=float, default=0.08, help="score gap below which a group counts as close")
    s.set_defaults(func=cmd_scan)

    q = sub.add_parser("lowq", parents=[common], help="standalone low-quality photos")
    q.add_argument("--topiq-max", type=float, default=0.30, help="flag if TOPIQ technical score is below (0.30)")
    q.add_argument("--aes-max", type=float, default=-0.30, help="flag if Apple aesthetic score is below (-0.30)")
    q.add_argument("--vlm", action="store_true", help="ask the local VLM to confirm each flagged photo")
    q.set_defaults(func=cmd_lowq)

    ln = sub.add_parser("learn", parents=[lib], help="learn weights from your deletion decisions")
    ln.set_defaults(func=cmd_learn)

    v = sub.add_parser("serve", parents=[lib], help="web UI to review a report and approve deletions")
    v.add_argument("report", nargs="?", help="report CSV path or file name (default: the newest report)")
    v.add_argument("--port", type=int, default=8765)
    v.add_argument("--no-open", action="store_true", help="don't open the browser")
    v.set_defaults(func=cmd_serve)

    args = ap.parse_args(argv)
    if sys.platform != "darwin":
        sys.exit("This script only runs on macOS (Photos library).")
    global LIBRARY
    LIBRARY = args.library
    args.func(args)


if __name__ == "__main__":
    main()
