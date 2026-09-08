"""Form parameters for the modifiers routes."""

from typing import Annotated

from fastapi import Form


class EndBeadParams:
    """The `new_end_type`/`base_type` pair the rewrite-end-beads route takes."""

    def __init__(
        self,
        new_end_type: Annotated[int, Form(ge=1)],
        base_type: Annotated[int, Form(ge=1)] = 1,
    ) -> None:
        self.new_end_type = new_end_type
        self.base_type = base_type
