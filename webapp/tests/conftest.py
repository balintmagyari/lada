from pathlib import Path

import pytest
from starlette.testclient import TestClient

from lada_web.config import Settings
from lada_web.main import create_app

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "tests" / "data"


@pytest.fixture
def data_dir() -> Path:
    assert DATA_DIR.is_dir(), f"Root fixtures directory missing: {DATA_DIR}"
    return DATA_DIR


@pytest.fixture
def temp_root(tmp_path: Path) -> Path:
    return tmp_path / "uploads"


@pytest.fixture
def settings(temp_root: Path) -> Settings:
    return Settings(
        max_upload_mb=1,
        max_trajectory_mb=1,
        max_trajectory_frames=100,
        max_table_rows=20,
        temp_root=temp_root,
    )


@pytest.fixture
def app(settings: Settings):
    return create_app(settings)


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def htmx() -> dict[str, str]:
    return {"HX-Request": "true"}


def upload(path: Path) -> dict:
    """files= argument for TestClient.post."""
    return {"file": (path.name, path.read_bytes())}
