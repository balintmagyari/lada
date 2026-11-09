from pathlib import Path

import pytest
from starlette.testclient import TestClient

from lada_web.config import Settings
from lada_web.main import create_app

RHEO_DATA = Path(__file__).resolve().parent / "data" / "rheometry"


@pytest.fixture
def rheo_data() -> Path:
    return RHEO_DATA


@pytest.fixture
def temp_root(tmp_path: Path) -> Path:
    return tmp_path / "uploads"


@pytest.fixture
def settings(temp_root: Path) -> Settings:
    return Settings(max_upload_mb=1, temp_root=temp_root)


@pytest.fixture
def app(settings: Settings):
    return create_app(settings)


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def files_upload(*paths: Path) -> list:
    """files= argument for TestClient.post: one or more files under the 'files' key, the
    field the import route reads."""
    return [("files", (p.name, p.read_bytes())) for p in paths]
