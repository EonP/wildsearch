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
) -> None:
    """Search a folder with a free-text query."""
    from wildsearch.clip import ClipEmbedder  # slow import, so only when searching

    folder = in_path or Path.cwd()

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
