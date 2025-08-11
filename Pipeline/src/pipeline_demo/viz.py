from __future__ import annotations

from pathlib import Path
from typing import Optional

from sklearn.utils import estimator_html_repr

from .pipeline import build_pipeline


def export_pipeline_html(outfile: Optional[str] = None) -> str:
    """Export the demo pipeline diagram to an HTML file.

    Returns the output file path written.
    """
    pipe = build_pipeline()
    html = estimator_html_repr(pipe)

    out = Path(outfile or "pipeline_demo.html")
    out.write_text(html, encoding="utf-8")
    return str(out)


def cli_viz(argv: Optional[list[str]] = None) -> int:
    """Console entry point: write pipeline_demo.html in the current directory.

    Usage: pipeline-demo-viz [--out FILE]
    """
    import argparse

    parser = argparse.ArgumentParser(description="Export pipeline HTML diagram.")
    parser.add_argument("--out", type=str, default="pipeline_demo.html", help="Output HTML file path.")
    args = parser.parse_args(argv)

    out_path = export_pipeline_html(args.out)
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_viz())
