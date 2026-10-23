import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from wildsearch.search import search_folder

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Find photos by describing them in plain language."""
    # Empty on purpose: a callback keeps Typer in "group of commands" mode, so the
    # command stays `wildsearch search ...` even while `search` is the only command.


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Plain-language description of the photo.")],
    n: Annotated[int, typer.Option("-n", min=1, help="Number of results to show.")] = 10,
    in_path: Annotated[
        Path | None,
        typer.Option(
            "-p",
            exists=True,
            file_okay=False,
            help="Folder to search. Defaults to the current folder.",
        ),
    ] = None,
    open_results: Annotated[
        bool,
        typer.Option("--open", help="Open the results together in Preview (macOS)."),
    ] = False,
) -> None:
    """Search a folder with a free-text query."""
    if open_results and sys.platform != "darwin":
        typer.echo("--open only works on macOS for now.", err=True)
        raise typer.Exit(code=2)
    folder = in_path or Path.cwd()

    from wildsearch.clip import ClipEmbedder  # slow import, so only when searching

    # Status messages go to stderr, so piping the results stays clean.
    typer.echo("Loading model…", err=True)
    embedder = ClipEmbedder()
    typer.echo(f"Searching {folder}…", err=True)
    results = search_folder(embedder, folder, query, n)

    if not results:
        typer.echo(f"No photos found in {folder}", err=True)
        raise typer.Exit(code=1)

    # One result per line on stdout: rank, score, path.
    for rank_number, (path, score) in enumerate(results, start=1):
        typer.echo(f"{rank_number:>3}  {score:.3f}  {path}")

    if open_results:
        open_in_preview([path for path, _ in results])


def open_in_preview(paths: list[Path]) -> None:
    """Open all photos in one Preview window, in the given order.

    A single `open` call with every file is what keeps them in one window with a
    sidebar; one call per file would open a window each.
    """
    files = [str(p) for p in paths]
    subprocess.run(["open", "-a", "Preview", *files], check=True)
