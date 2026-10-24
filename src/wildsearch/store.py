import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id       INTEGER PRIMARY KEY,
    path     TEXT    NOT NULL UNIQUE,
    size     INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    model_id TEXT    NOT NULL,
    vector   BLOB    NOT NULL,
    PRIMARY KEY (photo_id, model_id)
);
CREATE TABLE IF NOT EXISTS folders (
    path       TEXT    PRIMARY KEY,
    indexed_at INTEGER NOT NULL
);
"""


@dataclass(frozen=True)
class StoredPhoto:
    id: int
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class IndexedFolder:
    path: Path
    indexed_at: int  # Unix time, seconds
    photos: int
    embedded: int  # photos with a vector for at least one model


class Store:
    """The index database: photos and their vectors, in one SQLite file."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def commit(self) -> None:
        """Save everything since the last commit."""
        self._conn.commit()

    def add_photo(self, path: Path, size: int, mtime_ns: int) -> int:
        """Remember a new photo and return its id."""
        cur = self._conn.execute(
            "INSERT INTO photos (path, size, mtime_ns) VALUES (?, ?, ?)",
            (str(path), size, mtime_ns),
        )
        return cur.lastrowid

    def photos_in(self, folder: Path) -> dict[Path, StoredPhoto]:
        """Every stored photo inside `folder` (including subfolders)."""
        prefix = _folder_prefix(folder)  # adds a trailing "/"
        rows = self._conn.execute(
            "SELECT id, path, size, mtime_ns FROM photos WHERE substr(path, 1, ?) = ?",
            (len(prefix), prefix),
        )
        return {Path(p): StoredPhoto(i, s, m) for i, p, s, m in rows}

    def update_photo(self, photo_id: int, size: int, mtime_ns: int) -> None:
        """Record a changed file and forget its old vectors."""
        self._conn.execute(
            "UPDATE photos SET size = ?, mtime_ns = ? WHERE id = ?",
            (size, mtime_ns, photo_id),
        )
        self._conn.execute(
            "DELETE FROM embeddings WHERE photo_id = ?",
            (photo_id,),
        )

    def delete_photos(self, photo_ids: list[int]) -> None:
        """Forget photos that no longer exist. Their vectors go too (ON DELETE CASCADE)."""
        self._conn.executemany(
            "DELETE FROM photos WHERE id = ?",
            [(i,) for i in photo_ids],
        )

    def add_vectors(self, model_id: str, photo_ids: list[int], vectors: np.ndarray) -> None:
        """Save one vector per photo. Row i of `vectors` belongs to photo_ids[i]."""
        self._conn.executemany(
            "INSERT OR REPLACE INTO embeddings (photo_id, model_id, vector) VALUES (?, ?, ?)",
            [
                (photo_id, model_id, vec.astype("<f4").tobytes())
                for photo_id, vec in zip(photo_ids, vectors, strict=True)
            ],
        )

    def pending(self, model_id: str, folder: Path) -> list[tuple[int, Path]]:
        """Photos in `folder` that don't have a vector for `model_id` yet."""
        prefix = _folder_prefix(folder)
        rows = self._conn.execute(
            """
            SELECT id, path FROM photos
            WHERE substr(path, 1, ?) = ?
              AND NOT EXISTS (
                SELECT 1 FROM embeddings
                WHERE embeddings.photo_id = photos.id AND embeddings.model_id = ?
              )
            ORDER BY path
            """,
            (len(prefix), prefix, model_id),
        )
        return [(i, Path(p)) for i, p in rows]

    def indexed_paths(self, model_id: str, folder: Path) -> dict[Path, StoredPhoto]:
        """Photos in `folder` that have a vector for `model_id`, without the vectors."""
        prefix = _folder_prefix(folder)
        rows = self._conn.execute(
            """
            SELECT photos.id, photos.path, photos.size, photos.mtime_ns
            FROM embeddings
            JOIN photos ON photos.id = embeddings.photo_id
            WHERE embeddings.model_id = ? AND substr(photos.path, 1, ?) = ?
            """,
            (model_id, len(prefix), prefix),
        )
        return {Path(p): StoredPhoto(i, s, m) for i, p, s, m in rows}

    def add_folder(self, folder: Path) -> None:
        """Remember that `folder` was indexed, now.

        Folders inside it are dropped from the list (the new one covers them), and
        nothing is added if a folder that contains it is already listed.
        """
        folder = folder.resolve()
        now = int(time.time())
        for (path,) in self._conn.execute("SELECT path FROM folders").fetchall():
            listed = Path(path)
            if folder == listed or folder.is_relative_to(listed):
                self._conn.execute("UPDATE folders SET indexed_at = ? WHERE path = ?", (now, path))
                return
        prefix = _folder_prefix(folder)
        self._conn.execute(
            "DELETE FROM folders WHERE substr(path || '/', 1, ?) = ?",
            (len(prefix), prefix),
        )
        self._conn.execute(
            "INSERT INTO folders (path, indexed_at) VALUES (?, ?)", (str(folder), now)
        )

    def folders(self) -> list[IndexedFolder]:
        """Every indexed folder, with its number of photos and of embedded photos."""
        listed = self._conn.execute("SELECT path, indexed_at FROM folders ORDER BY path").fetchall()
        result = []
        for path, indexed_at in listed:
            prefix = _folder_prefix(Path(path))
            photos, embedded = self._conn.execute(
                """
                SELECT COUNT(*),
                       COUNT(*) FILTER (WHERE EXISTS (
                           SELECT 1 FROM embeddings WHERE photo_id = photos.id))
                FROM photos WHERE substr(path, 1, ?) = ?
                """,
                (len(prefix), prefix),
            ).fetchone()
            result.append(IndexedFolder(Path(path), indexed_at, photos, embedded))
        return result

    def forget_folder(self, folder: Path) -> int:
        """Remove every photo in `folder` and its vectors. Returns how many photos.

        Also drops `folder` and any folder inside it from the list of indexed folders.
        """
        prefix = _folder_prefix(folder)
        cur = self._conn.execute(
            "DELETE FROM photos WHERE substr(path, 1, ?) = ?", (len(prefix), prefix)
        )
        self._conn.execute(
            "DELETE FROM folders WHERE substr(path || '/', 1, ?) = ?",
            (len(prefix), prefix),
        )
        return cur.rowcount

    def total_photos(self) -> int:
        """Number of photos in the whole index."""
        return self._conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]

    def load_vectors(self, model_id: str, folder: Path) -> tuple[list[Path], np.ndarray]:
        """All vectors of `model_id` in `folder`. Row i belongs to paths[i]."""

        prefix = _folder_prefix(folder)
        rows = self._conn.execute(
            """
                SELECT photos.path, embeddings.vector
                FROM embeddings
                JOIN photos ON photos.id = embeddings.photo_id
                WHERE embeddings.model_id = ? AND substr(photos.path, 1, ?) = ?
                ORDER BY photos.path
                """,
            (model_id, len(prefix), prefix),
        ).fetchall()
        if not rows:
            return [], np.empty((0, 0), dtype=np.float32)
        paths = [Path(path) for path, _ in rows]
        blob = b"".join(vector for _, vector in rows)
        matrix = np.frombuffer(blob, dtype="<f4").reshape(len(rows), -1)
        return paths, matrix


def _folder_prefix(folder: Path):
    return os.path.join(str(folder.resolve()), "")
