"""Shared form fields for the chain-conformation routes."""

from typing import Annotated

from fastapi import Form


def parse_coord_cols(text: str) -> list[str]:
    names = [part.strip() for part in text.split(",") if part.strip()]
    if len(names) != 3:
        raise ValueError(
            "Coordinate columns must be three comma-separated names, for example xu,yu,zu."
        )
    return names


class ChainColumns:
    """Column names shared by the conformation routes (flat multipart form fields)."""

    def __init__(
        self,
        coord_cols: Annotated[str, Form()] = "xu,yu,zu",
        molecule_col: Annotated[str, Form()] = "mol",
        atom_id_col: Annotated[str, Form()] = "id",
        mass_col: Annotated[str, Form()] = "",
    ) -> None:
        self.coord_cols = parse_coord_cols(coord_cols)
        self.molecule_col = molecule_col.strip()
        self.atom_id_col = atom_id_col.strip()
        self.mass_col = mass_col.strip() or None
