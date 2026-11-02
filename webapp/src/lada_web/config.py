"""Runtime settings, read once per application and injected into every route."""

from pathlib import Path

from fastapi import Request
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime limits. Every field can be overridden with a LADA_WEB_* environment variable."""

    model_config = SettingsConfigDict(env_prefix="LADA_WEB_", extra="ignore")

    max_upload_mb: float = 50
    max_trajectory_mb: float = 500
    max_trajectory_frames: int = 3000
    # A compressed .npz says nothing about its decompressed size: a few megabytes on the wire
    # can declare an array of hundreds of gigabytes. This caps what the shape is allowed to
    # ask for once loaded, independently of the upload size and the frame count.
    max_trajectory_memory_mb: float = 2048
    # A data file read for its chain topology. Far below max_upload_mb on purpose: the parser
    # reads the whole file into a list of strings and then parses that list, so peak memory is
    # several times the file size and it is all spent on a request thread.
    max_topology_mb: float = 20
    max_table_rows: int = 500
    # Repeat-seed files or regime files averaged/merged into one calculation.
    max_files_per_request: int = 10
    # Background jobs: how long a finished/abandoned job's result stays fetchable, and how many
    # calculations may run on background threads at once.
    job_ttl_seconds: float = 600
    max_concurrent_jobs: int = 4
    # Rheometer imports: data points across all blocks of one file, and how large an .xlsx may
    # unpack to (a workbook is a zip archive, so its upload size says little about its contents).
    max_rheometry_rows: int = 200_000
    max_xlsx_uncompressed_mb: float = 200
    temp_root: Path | None = None

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def max_xlsx_uncompressed_bytes(self) -> int:
        return int(self.max_xlsx_uncompressed_mb * 1024 * 1024)

    @property
    def max_topology_bytes(self) -> int:
        return int(self.max_topology_mb * 1024 * 1024)

    @property
    def max_trajectory_bytes(self) -> int:
        return int(self.max_trajectory_mb * 1024 * 1024)

    @property
    def max_trajectory_memory_bytes(self) -> int:
        return int(self.max_trajectory_memory_mb * 1024 * 1024)


def get_settings(request: Request) -> Settings:
    return request.app.state.settings
