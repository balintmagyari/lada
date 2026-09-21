import io
from pathlib import Path

import pytest
from starlette.datastructures import UploadFile

from lada_web.config import Settings
from lada_web.errors import UploadTooLarge
from lada_web.uploads import download_workspace, safe_filename, save_upload, save_uploads

MIB = 1024 * 1024


def make_upload(data: bytes, filename: str, *, size: int | None) -> UploadFile:
    return UploadFile(io.BytesIO(data), filename=filename, size=size)


def test_save_upload_writes_the_file(tmp_path: Path):
    data = b"Step Temp\n0 300\n"
    dest = save_upload(make_upload(data, "sample.txt", size=len(data)), tmp_path, MIB)
    assert dest == tmp_path / "sample.txt"
    assert dest.read_bytes() == data


def test_save_upload_rejects_declared_oversize(tmp_path: Path):
    data = b"x" * (MIB + MIB // 2)
    upload = make_upload(data, "big.txt", size=len(data))
    with pytest.raises(UploadTooLarge):
        save_upload(upload, tmp_path, MIB)
    assert list(tmp_path.iterdir()) == []


def test_save_upload_rejects_oversize_while_streaming(tmp_path: Path):
    data = b"x" * (MIB + MIB // 2)
    upload = make_upload(data, "big.txt", size=None)
    with pytest.raises(UploadTooLarge):
        save_upload(upload, tmp_path, MIB)
    assert list(tmp_path.iterdir()) == []


def test_upload_too_large_message_names_the_limit(tmp_path: Path):
    data = b"x" * (MIB + 1)
    upload = make_upload(data, "big.txt", size=len(data))
    with pytest.raises(UploadTooLarge, match="1 MiB upload limit"):
        save_upload(upload, tmp_path, MIB)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd", "passwd"),
        ("", "upload"),
        ("my file (1).npz", "my_file_1_.npz"),
    ],
)
def test_safe_filename(raw: str, expected: str):
    assert safe_filename(raw) == expected


def test_save_uploads_avoids_same_name_collisions(tmp_path: Path):
    a = make_upload(b"first", "trajectory.npz", size=5)
    b = make_upload(b"second", "trajectory.npz", size=6)
    paths = save_uploads([a, b], tmp_path, MIB, max_files=10)
    assert [p.name for p in paths] == ["trajectory.npz", "trajectory.npz"]
    assert paths[0] != paths[1]
    assert paths[0].read_bytes() == b"first"
    assert paths[1].read_bytes() == b"second"


def test_save_uploads_rejects_more_than_the_file_limit(tmp_path: Path):
    uploads = [make_upload(b"x", f"f{i}.npz", size=1) for i in range(3)]
    with pytest.raises(ValueError, match="At most 2 files"):
        save_uploads(uploads, tmp_path, MIB, max_files=2)


def test_save_uploads_rejects_an_empty_list(tmp_path: Path):
    with pytest.raises(ValueError, match="At least one file"):
        save_uploads([], tmp_path, MIB, max_files=10)


def test_save_uploads_enforces_the_per_file_size_limit(tmp_path: Path):
    small = make_upload(b"ok", "small.npz", size=2)
    big = make_upload(b"x" * (MIB + 1), "big.npz", size=MIB + 1)
    with pytest.raises(UploadTooLarge):
        save_uploads([small, big], tmp_path, MIB, max_files=10)


def test_download_workspace_keeps_the_directory_on_success(tmp_path: Path):
    settings = Settings(temp_root=tmp_path / "uploads")
    with download_workspace(settings) as ws:
        (ws / "out.csv").write_text("a,b\n")
    assert ws.is_dir()
    assert (ws / "out.csv").exists()


def test_download_workspace_removes_the_directory_on_error(tmp_path: Path):
    settings = Settings(temp_root=tmp_path / "uploads")
    with pytest.raises(ValueError, match="nope"):  # noqa: SIM117
        with download_workspace(settings) as ws:
            (ws / "out.csv").write_text("a,b\n")
            raise ValueError("nope")
    assert not ws.exists()
