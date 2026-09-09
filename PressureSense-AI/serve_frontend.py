from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    directory = Path(__file__).resolve().parent / "frontend"
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    address = ("127.0.0.1", 8080)
    print("Piezo dashboard: http://127.0.0.1:8080")
    ThreadingHTTPServer(address, handler).serve_forever()


if __name__ == "__main__":
    main()
