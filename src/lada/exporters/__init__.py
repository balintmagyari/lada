"""LaDa exporters module: Write analysis results to files for external tools.

Currently provides LaTeX/pgfplots export via write_pgfplots_table.
"""

from .latex_exporter import write_pgfplots_table

__all__ = ["write_pgfplots_table"]
