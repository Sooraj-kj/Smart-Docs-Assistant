import argparse
import os

import uvicorn

from smart_docs.api import app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Smart Document Assistant API.")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument(
        "--reload",
        action="store_true",
        default=os.getenv("RELOAD", "false").lower() == "true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
