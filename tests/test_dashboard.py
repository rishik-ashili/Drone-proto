import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from gdnav.dashboard.server import SimApp, make_handler


def test_dashboard_endpoints():
    app = SimApp("hello_square", log=False, speed=4.0)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        world = json.load(urllib.request.urlopen(base + "/api/world"))
        assert world["mission"]["name"] == "hello_square"
        assert b"<html" in urllib.request.urlopen(base + "/").read()
        req = urllib.request.Request(base + "/api/command", data=b'{"cmd":"start"}', method="POST",
                                     headers={"Content-Type": "application/json"})
        assert json.load(urllib.request.urlopen(req))["ok"]
        import time
        time.sleep(1.0)
        state = json.load(urllib.request.urlopen(base + "/api/state"))
        assert state["state"] != "IDLE"
    finally:
        httpd.shutdown()
