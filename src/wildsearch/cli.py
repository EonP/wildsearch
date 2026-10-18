from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    help="Find photos by describing them in plain language.",
    no_args_is_help=True,
)

DbOption = Annotated[
    Path | None,
    typer.Option(
        "--db",
        envvar="WILDSEARCH_DB",
        help="Path to the index database. Defaults to the user data directory.",
    ),
]


@app.command()
def index(
    paths: Annotated[list[Path], typer.Argument(help="Folders to scan for photos.")],
    db: DbOption = None,
) -> None:
    """Scan folders and embed new or changed photos."""
    typer.echo("index: not implemented yet")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Plain-language description of the photo.")],
    n: Annotated[int, typer.Option("-n", help="Number of results to show.")] = 10,
    in_path: Annotated[
        Path | None, typer.Option("-p", help="Only search within this folder.")
    ] = None,
    db: DbOption = None,
) -> None:
    """Search the index with a free-text query."""
    typer.echo("search: not implemented yet")


@app.command()
def status(db: DbOption = None) -> None:
    """Show indexed folders, photo counts and database size."""
    typer.echo("status: not implemented yet")
