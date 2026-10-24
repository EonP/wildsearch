"""Bring the index of a folder up to date: scan, compare with the store, embed."""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import batched
from pathlib import Path

from wildsearch.embedder import Embedder
from wildsearch.images import load_image
from wildsearch.scanner import scan
from wildsearch.store import Store


@dataclass(frozen=True)
class SyncReport:
    added: int
    changed: int
    removed: int


def sync_folder(store: Store, folder: Path) -> SyncReport:
    """Make the store's list of photos match what is on disk now.

    New files are added, changed files are updated (which drops their old vectors),
    and files that are gone are removed. No embedding happens here.
    """
    on_disk = {f.path: f for f in scan(folder)}
    stored = store.photos_in(folder)

    added = changed = 0
    for path, file in on_disk.items():
        old = stored.get(path)
        if old is None:
            store.add_photo(path, file.size, file.mtime_ns)
            added += 1
        elif (old.size, old.mtime_ns) != (file.size, file.mtime_ns):
            store.update_photo(old.id, file.size, file.mtime_ns)
            changed += 1

    gone = [old.id for path, old in stored.items() if path not in on_disk]
    store.delete_photos(gone)

    store.commit()
    return SyncReport(added=added, changed=changed, removed=len(gone))


def embed_pending(
    store: Store,
    embedder: Embedder,
    folder: Path,
    batch_size: int = 32,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Embed every photo in `folder` that has no vector yet. Returns how many.

    Commits after each batch, so an interruption loses at most one batch; the next
    run picks up the rest, because `pending` only returns photos without a vector.
    `on_progress(done, total)` is called once before starting and after each batch,
    so a caller can show progress without this module knowing about the terminal.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    todo = store.pending(embedder.model_id, folder)
    done = 0
    if on_progress:
        on_progress(done, len(todo))
    for batch in batched(todo, batch_size):
        ids = [photo_id for photo_id, _ in batch]
        images = [load_image(path) for _, path in batch]
        store.add_vectors(embedder.model_id, ids, embedder.embed_images(images))
        store.commit()
        done += len(batch)
        if on_progress:
            on_progress(done, len(todo))
    return done


@dataclass(frozen=True)
class FolderStatus:
    on_disk: int  # photos in the folder right now
    up_to_date: int  # of those, indexed and unchanged since
    deleted: int  # indexed, but the file is gone

    @property
    def not_indexed(self) -> int:
        """New or changed photos that a search would miss."""
        return self.on_disk - self.up_to_date


def folder_status(store: Store, folder: Path, model_id: str) -> FolderStatus:
    """How much of `folder` is covered by the index. Reads only; changes nothing."""
    on_disk = {f.path: f for f in scan(folder)}
    indexed = store.indexed_paths(model_id, folder)
    up_to_date = sum(
        1
        for path, file in on_disk.items()
        if (old := indexed.get(path)) is not None
        and (old.size, old.mtime_ns) == (file.size, file.mtime_ns)
    )
    deleted = sum(1 for path in indexed if path not in on_disk)
    return FolderStatus(on_disk=len(on_disk), up_to_date=up_to_date, deleted=deleted)
