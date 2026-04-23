"""Launch the Gradio demo as a standalone server (no FastAPI wrapper).

This is the fastest way to spin up the UI during development:
    python scripts/run_demo.py

For production, the demo is mounted inside the FastAPI app at /demo and
starts automatically with docker compose up extractor-api.

Flags:
    --port   Server port (default: 7860)
    --share  Generate a public Gradio link (tunnelled via gradio.live)
    --debug  Enable Gradio hot-reload
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Generate a public gradio.live URL")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    try:
        from demo.app import demo
    except ImportError as e:
        print(f"✗ Cannot import demo: {e}")
        print("  Make sure gradio is installed: pip install gradio")
        sys.exit(1)

    print(f"Starting EXTRACTOR demo on http://localhost:{args.port}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_api=False,
    )


if __name__ == "__main__":
    main()
