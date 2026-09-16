"""The display rounding: six significant figures on screen, exact bytes in a download."""

import re

import pytest

from lada_web.rendering import sigfig, templates

from .conftest import files_upload

RG_URL = "/api/conformations/rg"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # The value the Rg² sample produces: full precision on the wire, rounded on screen.
        (20.567500000000003, "20.5675"),
        # Timesteps are floats. A plain '%.6g' would print '2000' here but '1e+07' below,
        # and a timestep must never be rendered in scientific notation.
        (2000.0, "2000"),
        (10000000.0, "10000000"),
        (0.5, "0.5"),
        (1 / 3, "0.333333"),
        # Non-integral magnitudes far from 1 are what scientific notation is for.
        (1e-17, "1e-17"),
        (123456789.123456789, "1.23457e+08"),
        # Integers stay integers, strings pass through, None is an empty cell.
        (0, "0"),
        (None, ""),
        ("id type xu yu zu", "id type xu yu zu"),
    ],
)
def test_sigfig_rounds_for_display(value, expected):
    assert sigfig(value) == expected


def test_sigfig_never_turns_a_large_timestep_into_scientific_notation():
    """The regression this filter exists to avoid: 1e7 is a plausible timestep."""
    for exponent in range(3, 16):
        rendered = sigfig(float(10**exponent))
        assert "e" not in rendered, rendered
        assert rendered == str(10**exponent)


def test_sigfig_is_registered_as_a_template_filter():
    assert templates.env.filters["sigfig"] is sigfig
    assert templates.env.globals["significant_figures"] == 6


def test_table_shows_rounded_values_while_the_csv_download_stays_exact(client, data_dir, htmx):
    """One calculation rendered three ways: the screen rounds, JSON and the file do not."""
    path = data_dir / "sample_dump.dump"

    exact = client.post(RG_URL, files=files_upload(path))
    assert exact.status_code == 200, exact.text
    assert exact.json()["rows"][0] == [0.0, 20.567500000000003]

    partial = client.post(RG_URL, files=files_upload(path), headers=htmx)
    assert partial.status_code == 200, partial.text
    assert re.findall(r"<td>([^<]*)</td>", partial.text) == [
        "0",
        "20.5675",
        "1000",
        "20.5675",
        "2000",
        "20.5675",
    ]
    # The plot reads its numbers from data-result, which keeps every digit.
    assert "20.567500000000003" in re.search(r'data-result="([^"]*)"', partial.text).group(1)

    download = client.post(f"{RG_URL}/download?fmt=csv", files=files_upload(path))
    assert download.status_code == 200, download.text
    assert download.text.splitlines()[1] == "0,20.567500000000003"


def test_an_untruncated_result_still_explains_the_rounding(client, data_dir, htmx):
    r = client.post(RG_URL, files=files_upload(data_dir / "sample_dump.dump"), headers=htmx)
    note = re.search(r'<p class="result-note">(.*?)</p>', r.text, re.S).group(1)
    assert "Values are shown to 6 significant figures; downloads carry full precision." in note
    assert "Showing the first" not in note
