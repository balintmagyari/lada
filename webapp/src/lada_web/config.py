"""Runtime settings, read once per application and injected into every route."""

from pathlib import Path

from fastapi import Request
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime limits. Every field can be overridden with a LADA_WEB_* environment variable."""

    model_config = SettingsConfigDict(env_prefix="LADA_WEB_", extra="ignore")

    # One imported file. Rheometer exports are small: a long frequency sweep is a few hundred
    # kilobytes, a multi-sheet workbook a few megabytes.
    max_upload_mb: float = 50
    # A JSON body (re-derivations, figures, superposition) carries already-parsed series, never
    # a raw file, so it gets its own small ceiling instead of inheriting the upload budget,
    # which would let one request make the server parse gigabytes.
    max_json_mb: float = 5
    # Files one import request may carry.
    max_files_per_request: int = 10
    # Data points across all blocks of one file, and how large an .xlsx may unpack to (a
    # workbook is a zip archive, so its upload size says little about its contents).
    max_rheometry_rows: int = 200_000
    max_xlsx_uncompressed_mb: float = 200
    temp_root: Path | None = None

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def max_json_bytes(self) -> int:
        return int(self.max_json_mb * 1024 * 1024)

    @property
    def max_xlsx_uncompressed_bytes(self) -> int:
        return int(self.max_xlsx_uncompressed_mb * 1024 * 1024)


def get_settings(request: Request) -> Settings:
    return request.app.state.settings
