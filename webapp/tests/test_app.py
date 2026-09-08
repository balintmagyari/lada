import re

from lada_web.catalog import CALCULATIONS


def test_index_lists_every_calculation(client):
    r = client.get("/")
    assert r.status_code == 200
    for calc in CALCULATIONS:
        assert calc.path in r.text
        assert calc.title in r.text


def test_reference_page_has_form_contract(client):
    r = client.get("/parsers/log")
    assert r.status_code == 200
    assert 'id="calc-form"' in r.text
    assert 'hx-post="/api/parsers/log/table"' in r.text
    assert 'id="results"' in r.text
    assert 'name="file"' in r.text
    assert 'hx-encoding="multipart/form-data"' in r.text
    form_tag = re.search(r"<form[^>]*>", r.text).group(0)
    assert "action=" not in form_tag


def test_unknown_page_is_404_json_and_html(client):
    r = client.get("/parsers/nope")
    assert r.status_code == 404
    assert r.json()["detail"] == "There is no page at this address."
    r = client.get("/nope/nope", headers={"Accept": "text/html"})
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("text/html")
    assert "There is no page at this address." in r.text


def test_static_assets_served(client):
    assert client.get("/static/css/app.css").status_code == 200
    assert client.get("/static/js/app.js").status_code == 200


def test_api_docs_available(client):
    assert client.get("/api/docs").status_code == 200


def test_settings_are_injected(app):
    assert app.state.settings.max_upload_mb == 1
