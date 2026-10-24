import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from platformdirs import user_data_path
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from wildsearch.embedder import Embedder
from wildsearch.indexer import (
    FolderStatus,
    embed_pending,
    folder_status,
    sync_folder,
)
from wildsearch.search import search_index
from wildsearch.store import Store

app = typer.Typer(
    help="Find photos by describing them in plain language.",
    no_args_is_help=True,
)

# Two consoles: results go to stdout (pipeable); everything for the human watching
# (spinners, progress, summaries, errors) goes to stderr.
out = Console()
err = Console(stderr=True)

DEFAULT_DB = user_data_path("wildsearch") / "index.db"

DbOption = Annotated[
    Path,
    typer.Option(
        "--db",
        envvar="WILDSEARCH_DB",
        dir_okay=False,
        help="Index database file.",
    ),
]


def load_embedder() -> Embedder:
    """Load the model behind a spinner. The import is slow, so it happens only here."""
    with err.status("Loading model…"):
        from wildsearch.clip import ClipEmbedder

        return ClipEmbedder()


@app.command()
def index(
    folders: Annotated[
        list[Path],
        typer.Argument(exists=True, file_okay=False, help="Folders to index (with subfolders)."),
    ],
    db: DbOption = DEFAULT_DB,
) -> None:
    """Index folders: embed new and changed photos, forget deleted ones."""
    embedder = load_embedder()
    with Store(db) as store:
        for folder in folders:
            store.add_folder(folder)
            with err.status(f"Scanning {folder}…"):
                report = sync_folder(store, folder)
            err.print(
                f"[bold]{folder}[/]: {report.added} new, {report.changed} changed, "
                f"{report.removed} removed"
            )

            embedded = embed_with_progress(store, embedder, folder)

            if embedded:
                err.print(f"[green]✓[/] Embedded {embedded} photos")
            else:
                err.print("[green]✓[/] Already up to date")


def embed_with_progress(store: Store, embedder: Embedder, folder: Path) -> int:
    """Run embed_pending with a progress bar on stderr."""
    with Progress(
        SpinnerColumn(),
        TextColumn("Embedding"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=err,
        transient=True,  # the bar disappears when done; the summary line stays
    ) as progress:
        task = progress.add_task("embed", total=None)

        def update(done: int, total: int) -> None:
            progress.update(task, completed=done, total=total)

        return embed_pending(store, embedder, folder, on_progress=update)


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
    db: DbOption = DEFAULT_DB,
) -> None:
    """Search indexed photos with a free-text query."""
    # Check before the slow part, so the user doesn't wait for the model first.
    if open_results and sys.platform != "darwin":  # "darwin" means macOS
        err.print("[red]--open only works on macOS for now.[/]")
        raise typer.Exit(code=2)

    folder = (in_path or Path.cwd()).resolve()
    embedder = load_embedder()
    with Store(db) as store:
        results = search_index(store, embedder, folder, query, n)
        coverage = folder_status(store, folder, embedder.model_id)

    if not results:
        err.print(f"[yellow]No indexed photos in {folder}.[/]")
        err.print(f"Run: [bold]wildsearch index {folder}[/]")
        raise typer.Exit(code=1)

    # Photos deleted since the last index would still rank; don't show dead paths.
    results = [(path, score) for path, score in results if path.exists()]

    if out.is_terminal:
        print_table(results, folder)
    else:
        # Piped or redirected: plain lines, easy for other tools to read.
        for rank_number, (path, score) in enumerate(results, start=1):
            print(f"{rank_number:>3}  {score:.3f}  {path}")

    print_coverage(coverage, folder)

    if open_results and results:
        open_in_preview([path for path, _ in results])


def print_coverage(status: FolderStatus, folder: Path) -> None:
    """After every search: how much of the folder was searched (on stderr)."""
    if not status.not_indexed and not status.deleted:
        err.print(f"[dim]Searched all {status.on_disk} photos in {folder}.[/]")
        return
    parts = []
    if status.not_indexed:
        parts.append(f"{status.not_indexed} new or changed photos were not searched")
    if status.deleted:
        parts.append(f"{status.deleted} indexed photos no longer exist")
    err.print(
        f"[yellow]Searched {status.up_to_date} of {status.on_disk} photos: {'; '.join(parts)}.[/]"
    )
    err.print(f"Run: [bold]wildsearch index {folder}[/]")


@app.command()
def status(db: DbOption = DEFAULT_DB) -> None:
    """Show which folders are indexed, and the database location and size."""
    if not db.exists():
        err.print("[yellow]Nothing indexed yet.[/] Run: [bold]wildsearch index FOLDER[/]")
        raise typer.Exit(code=1)
    with Store(db) as store:
        folders = store.folders()
        untracked = store.total_photos() - sum(f.photos for f in folders)

    out.print(f"[dim]Index:[/] {db}  [dim]({db.stat().st_size / 1e6:.1f} MB)[/]")
    if untracked:
        # Photos indexed before folders were recorded, or from a forgotten subfolder.
        out.print(
            f"[yellow]{untracked:,} photos are not under any listed folder.[/] "
            "Run [bold]wildsearch index FOLDER[/] again to list their folder, "
            "or [bold]wildsearch reset[/] to start over."
        )
    if not folders:
        out.print("No folders listed yet.")
        return
    table = Table(box=None, header_style="dim", pad_edge=False)
    table.add_column("Folder")
    table.add_column("Photos", justify="right")
    table.add_column("Embedded", justify="right")
    table.add_column("Last indexed", style="dim")
    for f in folders:
        # An external drive that's unplugged is still in the index, just not reachable.
        name = str(f.path) if f.path.exists() else f"{f.path} [dim](not found)[/]"
        embedded = f"{f.embedded:,}" if f.embedded == f.photos else f"[yellow]{f.embedded:,}[/]"
        when = datetime.fromtimestamp(f.indexed_at).strftime("%Y-%m-%d %H:%M")
        table.add_row(name, f"{f.photos:,}", embedded, when)
    out.print(table)


@app.command()
def forget(
    folders: Annotated[list[Path], typer.Argument(help="Folders to remove from the index.")],
    db: DbOption = DEFAULT_DB,
) -> None:
    """Remove folders from the index. The photos themselves are not touched."""
    with Store(db) as store:
        for folder in folders:
            removed = store.forget_folder(folder)
            store.commit()
            err.print(f"[green]✓[/] {folder.resolve()}: forgot {removed:,} photos")


@app.command()
def reset(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask for confirmation.")] = False,
    db: DbOption = DEFAULT_DB,
) -> None:
    """Delete the whole index. The photos themselves are not touched."""
    if not db.exists():
        err.print("Nothing to reset: no index yet.")
        return
    if not yes:
        typer.confirm(f"Delete the index at {db}?", abort=True)
    db.unlink()
    err.print("[green]✓[/] Index deleted")


def print_table(results: list[tuple[Path, float]], folder: Path) -> None:
    """Results as a table: file name highlighted and clickable, scores coloured."""
    best = results[0][1]
    table = Table(box=None, header_style="dim", pad_edge=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Score", justify="right")
    table.add_column("Photo")
    for rank_number, (path, score) in enumerate(results, start=1):
        # Scores only mean something relative to each other, so colour by distance
        # from the best match rather than by absolute value.
        colour = "green" if score >= best - 0.02 else "yellow" if score >= best - 0.05 else "dim"
        try:
            shown = path.relative_to(folder)
        except ValueError:
            shown = path
        name = f"[dim]{shown.parent}/[/]" if str(shown.parent) != "." else ""
        name += f"[bold][link={path.as_uri()}]{shown.name}[/link][/bold]"
        table.add_row(str(rank_number), f"[{colour}]{score:.3f}[/]", name)
    out.print(table)


def open_in_preview(paths: list[Path]) -> None:
    """Open all photos in one Preview window, in the given order.

    A single `open` call with every file is what keeps them in one window with a
    sidebar; one call per file would open a window each.
    """
    subprocess.run(["open", "-a", "Preview", *map(str, paths)], check=True)
