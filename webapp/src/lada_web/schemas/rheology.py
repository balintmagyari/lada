"""Shared multipart fields for the rheology view/download route pairs."""

from typing import Annotated, Literal

from fastapi import Form

from .regimes import MAX_PRONY_MODES


class ThermoState:
    """System volume and temperature (LJ units), both required and strictly positive."""

    def __init__(
        self,
        volume: Annotated[float, Form(gt=0)],
        temperature: Annotated[float, Form(gt=0)],
    ) -> None:
        self.volume = volume
        self.temperature = temperature


class PronyParams:
    """Prony-series fit settings used to turn G(t) into G'(omega)/G''(omega)."""

    def __init__(
        self,
        method: Annotated[Literal["GK", "FSR", "both"], Form()] = "GK",
        t_min: Annotated[float, Form(gt=0)] = 1.0,
        t_cutoff: Annotated[float | None, Form()] = None,
        n_modes: Annotated[int, Form(ge=1, le=MAX_PRONY_MODES)] = 50,
        n_omega: Annotated[int, Form(ge=2, le=5000)] = 200,
        omega_min: Annotated[float | None, Form(gt=0)] = None,
        omega_max: Annotated[float | None, Form(gt=0)] = None,
    ) -> None:
        self.method = method
        self.t_min = t_min
        self.t_cutoff = t_cutoff
        self.n_modes = n_modes
        self.n_omega = n_omega
        self.omega_min = omega_min
        self.omega_max = omega_max
