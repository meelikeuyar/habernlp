"""Tests for the scraper module."""

import pytest
from unittest.mock import patch, MagicMock
from src.scraper.scraper import _tek_kaynak_cek, haber_cek


MOCK_HTML = """
<html><body>
<a href="/gundem/test-haberi-12345">Bu bir test haber başlığıdır ve yeterince uzundur</a>
<a href="/ekonomi/dolar-yukseldi-67890">Dolar yükseldi piyasalar çalkantılı bir gün yaşadı</a>
<a href="/kisa">Kısa</a>
<a href="/spor/mac-sonucu">Fenerbahçe Galatasaray derbisinde nefes kesen maç sonuçlandı</a>
</body></html>
"""

KAYNAK = {
    "ad": "Test Kaynak",
    "url": "https://test.com",
    "filtre": ["/gundem/", "/ekonomi/", "/spor/"],
}


@patch("src.scraper.scraper.requests.get")
def test_tek_kaynak_cek_parses_articles(mock_get):
    mock_resp = MagicMock()
    mock_resp.text = MOCK_HTML
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    result = _tek_kaynak_cek(KAYNAK)

    assert len(result) == 3
    assert all("baslik" in r for r in result)
    assert all("url" in r for r in result)
    assert all(r["kaynak"] == "Test Kaynak" for r in result)


@patch("src.scraper.scraper.requests.get")
def test_tek_kaynak_filters_short_titles(mock_get):
    mock_resp = MagicMock()
    mock_resp.text = MOCK_HTML
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    result = _tek_kaynak_cek(KAYNAK)
    titles = [r["baslik"] for r in result]

    assert "Kısa" not in titles


@patch("src.scraper.scraper.requests.get")
def test_tek_kaynak_handles_errors(mock_get):
    mock_get.side_effect = Exception("Connection failed")

    result = _tek_kaynak_cek(KAYNAK)

    assert result == []


@patch("src.scraper.scraper.requests.get")
def test_tek_kaynak_deduplicates(mock_get):
    dup_html = """
    <html><body>
    <a href="/gundem/haber-1">Aynı başlık tekrar eden bir haber metnidir</a>
    <a href="/gundem/haber-2">Aynı başlık tekrar eden bir haber metnidir</a>
    </body></html>
    """
    mock_resp = MagicMock()
    mock_resp.text = dup_html
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    result = _tek_kaynak_cek(KAYNAK)

    assert len(result) == 1
