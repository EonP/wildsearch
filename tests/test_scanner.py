import os
from pathlib import Path

import pytest

from wildsearch.scanner import ScannedFile, scan


def names(directory):
    return sorted(f.path.name for f in scan(directory))


def test_finds_images_in_nested_folders(tmp_path):
    (tmp_path / "a.jpg").touch()
    (tmp_path / "sub" / "deep").mkdir(parents=True)
    (tmp_path / "sub" / "b.png").touch()
    (tmp_path / "sub" / "deep" / "c.webp").touch()

    assert names(tmp_path) == ["a.jpg", "b.png", "c.webp"]


def test_ignores_non_image_files(tmp_path):
    (tmp_path / "a.jpg").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "doc.pdf").touch()
    (tmp_path / "jpg").touch()

    assert names(tmp_path) == ["a.jpg"]


def test_extension_matching_is_case_insensitive(tmp_path):
    (tmp_path / "IMG_1.JPG").touch()
    (tmp_path / "IMG_2.Png").touch()

    assert names(tmp_path) == ["IMG_1.JPG", "IMG_2.Png"]


@pytest.mark.parametrize("ext", [".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"])
def test_supported_extensions(tmp_path, ext):
    (tmp_path / f"photo{ext}").touch()

    assert names(tmp_path) == [f"photo{ext}"]


def test_skips_hidden_files(tmp_path):
    (tmp_path / "a.jpg").touch()
    (tmp_path / ".hidden.jpg").touch()
    (tmp_path / "._a.jpg").touch()

    assert names(tmp_path) == ["a.jpg"]


def test_skips_hidden_folders(tmp_path):
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "thumb.jpg").touch()
    (tmp_path / "a.jpg").touch()

    assert names(tmp_path) == ["a.jpg"]


def test_directory_named_like_an_image_is_not_a_file(tmp_path):
    (tmp_path / "folder.png").mkdir()
    (tmp_path / "folder.png" / "inside.jpg").touch()

    assert names(tmp_path) == ["inside.jpg"]


def test_reports_size_and_mtime(tmp_path):
    photo = tmp_path / "a.jpg"
    photo.write_bytes(b"12345")
    os.utime(photo, ns=(1_000_000_000, 1_700_000_000_123_456_789))

    (found,) = scan(tmp_path)

    assert isinstance(found, ScannedFile)
    assert found.size == 5
    assert found.mtime_ns == 1_700_000_000_123_456_789


def test_paths_are_absolute_even_for_relative_input(tmp_path, monkeypatch):
    (tmp_path / "a.jpg").touch()
    monkeypatch.chdir(tmp_path)  # cd into the tmp folder

    (found,) = scan(tmp_path)
    (relative,) = scan(Path("."))

    assert found.path.is_absolute()
    assert relative.path.is_absolute()
    assert relative.path == (tmp_path / "a.jpg").resolve()


def test_empty_folder_yields_nothing(tmp_path):
    assert list(scan(tmp_path)) == []


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(scan(tmp_path / "nope"))


def test_file_instead_of_folder_raises(tmp_path):
    photo = tmp_path / "a.jpg"
    photo.touch()

    with pytest.raises(FileNotFoundError):
        list(scan(photo))


def test_symlink_loop_does_not_hang(tmp_path):
    (tmp_path / "a.jpg").touch()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "loop").symlink_to(tmp_path, target_is_directory=True)

    assert names(tmp_path) == ["a.jpg"]


def test_symlinked_root_is_followed(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "a.jpg").touch()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    (found,) = scan(link)

    assert found.path == (real / "a.jpg").resolve()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")
def test_unreadable_folder_is_skipped(tmp_path):
    (tmp_path / "a.jpg").touch()
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "b.jpg").touch()
    original_mode = locked.stat().st_mode
    locked.chmod(0o000)  # no read, write or execute for anyone
    try:
        assert names(tmp_path) == ["a.jpg"]
    finally:
        locked.chmod(original_mode)  # so pytest can clean up tmp_path
