"""Shared multipart fields for the parsers route pairs."""

from typing import Annotated

from fastapi import Form


class DataSection:
    """Which section of a LAMMPS data file to read."""

    def __init__(self, section: Annotated[str, Form()] = "Atoms") -> None:
        self.section = section
