"""Local web UI for reviewing a photo_cleanup report and approving deletions.

    uv run photo_cleanup.py serve [REPORT] [--library PATH] [--port 8765]

Binds to 127.0.0.1 only. Decisions are saved next to the report as `<report>.decisions.json`.
"Apply" tags the approved photos with the keyword `cleanup_delete` (and removes it from photos in this
report you no longer approve). Photos' scripting interface cannot delete photos, so the last step is
yours: smart album "Keyword is cleanup_delete" → select all → ⌘⌫ (they go to Recently Deleted).
"""
from __future__ import annotations

import csv
import json
import threading
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

DELETE_KEYWORD = "cleanup_delete"
UI_FILE = Path(__file__).with_name("review_ui.html")
THUMB_SIZES = {"s": 360, "l": 1600}
STATES = {"delete", "keep"}


# ---------------------------------------------------------------- reports

def report_kind(path: Path) -> str | None:
    try:
        with open(path, encoding="utf-8") as fh:
            header = fh.readline().strip().split(";")
    except OSError:
        return None
    if "verdict" in header:
        return "lowq"
    if "group" in header:
        return "scan"
    return None


def list_reports(report_dir: Path) -> list[dict]:
    out = []
    for p in sorted(report_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
        kind = report_kind(p)
        if kind:
            out.append({"name": p.name, "kind": kind,
                        "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")})
    return out


def _f(v):
    try:
        return None if v in (None, "") else round(float(v), 3)
    except ValueError:
        return None


def load_rows(path: Path) -> tuple[str, list[dict]]:
    kind = report_kind(path)
    items = []
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            if not r.get("uuid"):
                continue
            if kind == "scan":
                best = r.get("keep") == "yes"
                items.append({
                    "uuid": r["uuid"], "filename": r.get("filename", ""), "date": r.get("date", ""),
                    "group": int(r.get("group") or 0), "rank": int(r.get("rank") or 0), "best": best,
                    "suggestion": "keep" if best else "delete",
                    "reason": r.get("vlm_reason", ""), "flags": ["close"] if r.get("close") == "yes" else [],
                    "scores": {k: _f(r.get(k)) for k in ("score", "topiq", "laion", "apple_aes", "face_q", "eyes")},
                })
            else:
                verdict = (r.get("verdict") or "").lower()
                items.append({
                    "uuid": r["uuid"], "filename": r.get("filename", ""), "date": r.get("date", ""),
                    "suggestion": {"delete": "delete", "candidate": "delete", "keep": "keep"}.get(verdict, "unsure"),
                    "verdict": verdict, "reason": r.get("reason", ""),
                    "flags": [f for f in (r.get("flags") or "").split(",") if f],
                    "scores": {k: _f(r.get(k)) for k in ("topiq", "laion", "apple_aes")},
                    "_path": r.get("path") or None,
                })
    return kind, items


# ---------------------------------------------------------------- server state

class Review:
    def __init__(self, api, initial: Path | None):
        self.api = api
        self.report_dir = Path(api.REPORT_DIR)
        self.thumb_dir = Path(api.HOME) / "thumbs"
        external = initial and initial.parent.resolve() != self.report_dir.resolve()
        self.extra = {initial.name: initial} if external else {}
        self.initial = initial.name if initial else None
        self.lock = threading.Lock()
        self.apply_lock = threading.Lock()
        self.paths: dict[str, str] = {}

    def report_path(self, name: str | None) -> Path:
        name = name or self.initial
        if not name:
            raise FileNotFoundError("no reports found – run `scan` or `lowq` first")
        if name in self.extra:
            return self.extra[name]
        p = (self.report_dir / Path(name).name)
        if not p.exists() or not report_kind(p):
            raise FileNotFoundError(f"report not found: {name}")
        return p

    @staticmethod
    def decisions_path(report: Path) -> Path:
        return report.with_name(report.name + ".decisions.json")

    def read_decisions(self, report: Path) -> dict:
        p = self.decisions_path(report)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                pass
        return {"decisions": {}, "applied": None}

    def write_decisions(self, report: Path, data: dict) -> None:
        p = self.decisions_path(report)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(p)

    def report(self, name: str | None) -> dict:
        path = self.report_path(name)
        kind, items = load_rows(path)
        for it in items:
            if it.get("_path"):
                self.paths.setdefault(it["uuid"], it["_path"])
            it.pop("_path", None)
        d = self.read_decisions(path)
        return {"name": path.name, "kind": kind, "items": items,
                "decisions": d.get("decisions", {}), "applied": d.get("applied"),
                "keyword": DELETE_KEYWORD}

    def save(self, name: str, changes: dict) -> dict:
        path = self.report_path(name)
        with self.lock:
            d = self.read_decisions(path)
            dec = d.setdefault("decisions", {})
            for uuid, state in changes.items():
                if state in STATES:
                    dec[uuid] = state
                else:
                    dec.pop(uuid, None)
            self.write_decisions(path, d)
        return {"ok": True, "count": len(dec)}

    def apply(self, name: str) -> dict:
        path = self.report_path(name)
        _, items = load_rows(path)
        scope = {it["uuid"] for it in items}
        d = self.read_decisions(path)
        approved = sorted(u for u, s in d.get("decisions", {}).items() if s == "delete" and u in scope)
        if not self.apply_lock.acquire(blocking=False):
            return {"ok": False, "error": "an apply is already running"}
        try:
            if not self.api.check_photos_app_library():
                return {"ok": False, "error": "The Photos app has a different library open than --library "
                                              "(or none was given). Open the right library in Photos and retry."}
            res = self.api.tag_photos(DELETE_KEYWORD, approved, scope)
        except SystemExit as e:  # resolve_library() exits when the library is missing
            return {"ok": False, "error": str(e)}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        finally:
            self.apply_lock.release()
        with self.lock:
            d = self.read_decisions(path)
            d["applied"] = {"at": datetime.now().isoformat(timespec="seconds"), "count": len(approved), **res}
            self.write_decisions(path, d)
        return {"ok": True, "approved": len(approved), "keyword": DELETE_KEYWORD, **res}

    def photo_path(self, uuid: str) -> str | None:
        if uuid in self.paths:
            return self.paths[uuid]
        db = Path(self.api.DB_PATH)
        if db.exists():
            import sqlite3
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                row = con.execute("SELECT path FROM photos WHERE uuid=?", (uuid,)).fetchone()
            finally:
                con.close()
            if row and row[0]:
                self.paths[uuid] = row[0]
                return row[0]
        return None

    def thumb(self, uuid: str, size: str) -> bytes | None:
        px = THUMB_SIZES.get(size, THUMB_SIZES["s"])
        out = self.thumb_dir / size / f"{uuid}.jpg"
        if out.exists():
            return out.read_bytes()
        src = self.photo_path(uuid)
        if not src or not Path(src).exists():
            return None
        im = self.api.load_image(src, px)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{threading.get_ident()}.tmp")
        im.save(tmp, format="JPEG", quality=82)
        tmp.replace(out)
        return out.read_bytes()


# ---------------------------------------------------------------- HTTP

def make_handler(review: Review, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the terminal quiet except for errors
            if args and str(args[1]).startswith(("4", "5")):
                super().log_message(fmt, *args)

        def _send(self, status, body: bytes, ctype: str, cache: bool = False):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status=HTTPStatus.OK):
            self._send(status, json.dumps(obj).encode(), "application/json")

        def _guard(self) -> bool:
            # Block DNS rebinding: only answer requests addressed to this local server.
            if self.headers.get("Host") not in allowed_hosts:
                self._json({"error": "bad host"}, HTTPStatus.FORBIDDEN)
                return False
            return True

        def do_GET(self):
            if not self._guard():
                return
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    self._send(HTTPStatus.OK, UI_FILE.read_bytes(), "text/html; charset=utf-8")
                elif u.path == "/api/reports":
                    self._json({"reports": list_reports(review.report_dir) +
                                [{"name": n, "kind": report_kind(p), "mtime": ""} for n, p in review.extra.items()],
                                "initial": review.initial})
                elif u.path == "/api/report":
                    self._json(review.report(q.get("name")))
                elif u.path.startswith("/thumb/"):
                    uuid = unquote(u.path[len("/thumb/"):])
                    data = review.thumb(uuid, q.get("size", "s"))
                    if data is None:
                        self._send(HTTPStatus.NOT_FOUND, b"", "image/jpeg")
                    else:
                        self._send(HTTPStatus.OK, data, "image/jpeg", cache=True)
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except FileNotFoundError as e:
                self._json({"error": str(e)}, HTTPStatus.NOT_FOUND)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # noqa: BLE001
                self._json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self):
            if not self._guard():
                return
            # A JSON content type forces a CORS preflight, which this server never approves,
            # so other web pages can't post here.
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                self._json({"error": "JSON only"}, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
                return
            u = urlparse(self.path)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if u.path == "/api/decisions":
                    self._json(review.save(body["name"], body.get("changes", {})))
                elif u.path == "/api/apply":
                    self._json(review.apply(body["name"]))
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except FileNotFoundError as e:
                self._json({"error": str(e)}, HTTPStatus.NOT_FOUND)
            except Exception as e:  # noqa: BLE001
                self._json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.BAD_REQUEST)

    return Handler


def serve(api, args) -> None:
    report_dir = Path(api.REPORT_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)
    initial = None
    if args.report:
        p = Path(args.report).expanduser()
        if not p.exists():
            p = report_dir / args.report
        if not p.exists() or not report_kind(p):
            raise SystemExit(f"Not a photo_cleanup report: {args.report}")
        initial = p.resolve()
    else:
        reports = list_reports(report_dir)
        initial = report_dir / reports[0]["name"] if reports else None

    review = Review(api, initial)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(review, args.port))
    url = f"http://127.0.0.1:{args.port}/"
    api.log(f"Reviewing {initial.name if initial else '(no reports yet)'} at {url}  (Ctrl+C to stop)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
