import argparse
import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the forcingkit service")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=9598, help="Bind port")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes (default: 1)"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload (dev mode)"
    )

    args = parser.parse_args()

    if args.reload:
        print(f"Reload enabled: Forcing workers=1 on port {args.port}")
        uvicorn.run(
            "forcingkit_serve.main:app", host=args.host, port=args.port, reload=True
        )
    else:
        print(
            f"Starting Data Ingestor Service on {args.host}:{args.port} with {args.workers} workers."
        )
        uvicorn.run(
            "forcingkit_serve.main:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
