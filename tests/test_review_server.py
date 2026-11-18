import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

import review_server as rs

SCAN_CSV = (
    "group;rank;keep;uuid;filename;date;score;topiq;laion;apple_aes;face_q;eyes;close;vlm_reason\n"
    "1;1;yes;U1;a.jpg;2024-01-01;0.9;0.8;5.1;0.4;;;;\n"
    "1;2;;U2;b.jpg;2024-01-01;0.5;0.6;;;;;yes;blurry\n"
    ";;;;nouuid.jpg;;;;;;;;;\n"
)
LOWQ_CSV = (
    "verdict;uuid;filename;date;topiq;laion;apple_aes;flags;reason;path\n"
    "delete;L1;x.jpg;2024-02-01;0.1;2;0.1;blur,dark;bad;/p/x.jpg\n"
    "candidate;L2;y.jpg;2024-02-01;0.2;;;;;\n"
    "keep;L3;z.jpg;2024-02-01;0.9;;;;;\n"
    "weird;L4;w.jpg;2024-02-01;;;;;;\n"
)


@pytest.fixture
def env(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "scan.csv").write_text(SCAN_CSV)
    (reports / "lowq.csv").write_text(LOWQ_CSV)
    (reports / "other.csv").write_text("foo;bar\n1;2\n")
    calls = []

    def tag_photos(keyword, uuids, scope):
        calls.append((keyword, list(uuids), set(scope)))
        return {"tagged": len(uuids)}

    api = SimpleNamespace(
        REPORT_DIR=reports, HOME=tmp_path, DB_PATH=tmp_path / "cache.sqlite",
        check_photos_app_library=lambda: True, tag_photos=tag_photos,
        load_image=None, calls=calls,
    )
    return api, reports


# ---------------------------------------------------------------- report parsing

def test_report_kind(env):
    _, reports = env
    assert rs.report_kind(reports / "scan.csv") == "scan"
    assert rs.report_kind(reports / "lowq.csv") == "lowq"
    assert rs.report_kind(reports / "other.csv") is None
    assert rs.report_kind(reports / "missing.csv") is None


def test_list_reports_skips_non_reports(env):
    _, reports = env
    assert {r["name"]: r["kind"] for r in rs.list_reports(reports)} == {"scan.csv": "scan", "lowq.csv": "lowq"}


def test_load_scan_rows(env):
    _, reports = env
    kind, items = rs.load_rows(reports / "scan.csv")
    assert kind == "scan"
    assert [i["uuid"] for i in items] == ["U1", "U2"]  # row without uuid skipped
    u1, u2 = items
    assert (u1["best"], u1["suggestion"], u1["flags"]) == (True, "keep", [])
    assert (u2["best"], u2["suggestion"], u2["flags"], u2["reason"]) == (False, "delete", ["close"], "blurry")
    assert u1["scores"]["topiq"] == 0.8 and u2["scores"]["laion"] is None


def test_load_lowq_rows(env):
    _, reports = env
    kind, items = rs.load_rows(reports / "lowq.csv")
    assert kind == "lowq"
    assert [i["suggestion"] for i in items] == ["delete", "delete", "keep", "unsure"]
    assert items[0]["flags"] == ["blur", "dark"] and items[0]["_path"] == "/p/x.jpg"
    assert items[1]["_path"] is None


def test_f_parses_numbers():
    assert rs._f("0.12345") == 0.123
    assert rs._f("") is None and rs._f(None) is None and rs._f("abc") is None


# ---------------------------------------------------------------- Review state

def test_report_path_rejects_traversal_and_non_reports(env):
    api, reports = env
    r = rs.Review(api, None)
    assert r.report_path("scan.csv") == reports / "scan.csv"
    with pytest.raises(FileNotFoundError):
        r.report_path("other.csv")          # exists but not a report
    with pytest.raises(FileNotFoundError):
        r.report_path("nope.csv")
    with pytest.raises(FileNotFoundError):
        r.report_path(None)                 # no initial report
    secret = reports.parent / "secret.csv"
    secret.write_text(SCAN_CSV)
    with pytest.raises(FileNotFoundError):
        r.report_path("../secret.csv")      # directory components are stripped, so it is not found in reports/


def test_external_initial_report_is_served(env, tmp_path):
    api, _ = env
    ext = tmp_path / "elsewhere" / "ext.csv"
    ext.parent.mkdir()
    ext.write_text(SCAN_CSV)
    r = rs.Review(api, ext)
    assert r.report_path(None) == ext and r.report_path("ext.csv") == ext


def test_save_and_read_decisions(env):
    api, reports = env
    r = rs.Review(api, None)
    assert r.save("scan.csv", {"U1": "delete", "U2": "keep"}) == {"ok": True, "count": 2}
    assert r.save("scan.csv", {"U1": "clear", "U2": "bogus"})["count"] == 0  # invalid state clears the entry
    r.save("scan.csv", {"U2": "delete"})
    rep = r.report("scan.csv")
    assert rep["decisions"] == {"U2": "delete"} and rep["applied"] is None
    assert (reports / "scan.csv.decisions.json").exists()


def test_corrupt_decisions_file_is_tolerated(env):
    api, reports = env
    (reports / "scan.csv.decisions.json").write_text("{not json")
    assert rs.Review(api, None).report("scan.csv")["decisions"] == {}


def test_report_remembers_photo_paths_and_hides_them(env):
    api, _ = env
    r = rs.Review(api, None)
    rep = r.report("lowq.csv")
    assert all("_path" not in i for i in rep["items"])
    assert r.photo_path("L1") == "/p/x.jpg"
    assert r.photo_path("unknown") is None


def test_apply_only_tags_approved_photos_in_report(env):
    api, reports = env
    r = rs.Review(api, None)
    r.save("scan.csv", {"U1": "keep", "U2": "delete", "FOREIGN": "delete"})
    out = r.apply("scan.csv")
    assert out["ok"] and out["approved"] == 1 and out["keyword"] == rs.DELETE_KEYWORD
    keyword, uuids, scope = api.calls[0]
    assert uuids == ["U2"]                      # not the 'keep', not the uuid outside the report
    assert scope == {"U1", "U2"}
    assert r.report("scan.csv")["applied"]["count"] == 1


def test_apply_refuses_when_wrong_library_open(env):
    api, _ = env
    api.check_photos_app_library = lambda: False
    r = rs.Review(api, None)
    r.save("scan.csv", {"U2": "delete"})
    out = r.apply("scan.csv")
    assert out["ok"] is False and "library" in out["error"] and api.calls == []


def test_apply_reports_errors_and_releases_lock(env):
    api, _ = env

    def boom(*a):
        raise RuntimeError("osascript failed")
    api.tag_photos = boom
    r = rs.Review(api, None)
    out = r.apply("scan.csv")
    assert out["ok"] is False and "RuntimeError" in out["error"]
    assert r.apply_lock.acquire(blocking=False)  # lock was released


def test_apply_rejects_concurrent_run(env):
    api, _ = env
    r = rs.Review(api, None)
    r.apply_lock.acquire()
    assert r.apply("scan.csv") == {"ok": False, "error": "an apply is already running"}


def test_photo_path_falls_back_to_cache_db(env):
    import sqlite3
    api, _ = env
    con = sqlite3.connect(api.DB_PATH)
    con.execute("CREATE TABLE photos(uuid TEXT PRIMARY KEY, path TEXT)")
    con.execute("INSERT INTO photos VALUES('U9', '/db/path.jpg')")
    con.commit()
    con.close()
    assert rs.Review(api, None).photo_path("U9") == "/db/path.jpg"


# ---------------------------------------------------------------- HTTP

@pytest.fixture
def server(env):
    api, _ = env
    review = rs.Review(api, None)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), lambda *a: None)
    port = httpd.server_address[1]
    httpd.RequestHandlerClass = rs.make_handler(review, port)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield port, api
    httpd.shutdown()
    httpd.server_close()


def req(port, method, path, body=None, headers=None):
    h = {"Host": f"127.0.0.1:{port}"}
    h.update(headers or {})
    c = http.client.HTTPConnection("127.0.0.1", port)
    c.request(method, path, body=body, headers=h)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def post(port, path, obj, **kw):
    headers = {"Content-Type": "application/json", **kw.pop("headers", {})}
    return req(port, "POST", path, json.dumps(obj), headers)


def test_http_rejects_bad_host(server):
    port, _ = server
    for method in ("GET", "POST"):
        status, _ = req(port, method, "/api/reports", headers={"Host": "evil.example.com"})
        assert status == 403


def test_http_get_endpoints(server):
    port, _ = server
    status, data = req(port, "GET", "/api/reports")
    assert status == 200 and {r["name"] for r in json.loads(data)["reports"]} == {"scan.csv", "lowq.csv"}
    status, data = req(port, "GET", "/api/report?name=scan.csv")
    assert status == 200 and len(json.loads(data)["items"]) == 2
    assert req(port, "GET", "/api/report?name=nope.csv")[0] == 404
    assert req(port, "GET", "/api/report?name=../../etc/passwd")[0] == 404
    assert req(port, "GET", "/nothing")[0] == 404
    assert req(port, "GET", "/thumb/UNKNOWN")[0] == 404


def test_http_post_requires_json_content_type(server):
    port, _ = server
    status, _ = req(port, "POST", "/api/decisions", "{}", {"Content-Type": "text/plain"})
    assert status == 415


def test_http_save_and_apply_roundtrip(server):
    port, api = server
    status, data = post(port, "/api/decisions", {"name": "scan.csv", "changes": {"U2": "delete"}})
    assert status == 200 and json.loads(data)["count"] == 1
    status, data = post(port, "/api/apply", {"name": "scan.csv"})
    assert status == 200 and json.loads(data)["approved"] == 1
    assert api.calls[0][1] == ["U2"]


def test_http_post_errors(server):
    port, _ = server
    assert post(port, "/api/decisions", {"name": "nope.csv"})[0] == 404
    assert post(port, "/api/decisions", {})[0] == 400          # missing "name"
    assert post(port, "/api/other", {})[0] == 404
    assert req(port, "POST", "/api/apply", "{bad", {"Content-Type": "application/json"})[0] == 400
