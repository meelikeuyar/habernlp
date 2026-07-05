"""Hangi RSS feed'lerin çalıştığını test eder. Proje root'unda çalıştır:
   python test_feeds.py
"""

import requests
import feedparser

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    )
}

FEEDS = [
    ("T24 RSS",          "https://t24.com.tr/rss"),
    ("BBC Türkçe",       "https://feeds.bbci.co.uk/turkce/rss.xml"),
    ("NTV Gündem",       "https://www.ntv.com.tr/gundem.rss"),
    ("NTV Dünya",        "https://www.ntv.com.tr/dunya.rss"),
    ("Hürriyet Gündem",  "https://www.hurriyet.com.tr/rss/gundem"),
    ("Hürriyet Ekonomi", "https://www.hurriyet.com.tr/rss/ekonomi"),
    ("Sözcü Gündem",     "https://www.sozcu.com.tr/rss/gundem.xml"),
    ("Cumhuriyet",       "https://www.cumhuriyet.com.tr/rss/son-dakika.xml"),
    ("Habertürk",        "https://www.haberturk.com/rss"),
    ("TRT Haber",        "https://www.trthaber.com/manset.rss"),
    ("Euronews TR",      "https://tr.euronews.com/rss"),
    ("DW Türkçe",        "https://rss.dw.com/xml/rss-tur-all"),
    ("Independent TR",   "https://www.indyturk.com/rss.xml"),
]

print(f"{'Kaynak':<22} {'Status':<8} {'Haber':>6}  {'Örnek Başlık'}")
print("-" * 90)

calisan, calismayanlar = 0, []

for name, url in FEEDS:
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            print(f"{name:<22} {r.status_code:<8} {'—':>6}  HTTP error")
            calismayanlar.append((name, url, f"HTTP {r.status_code}"))
            continue
        d = feedparser.parse(r.text)
        n = len(d.entries)
        ornek = d.entries[0]["title"][:50] if d.entries else "—"
        status = "✓" if n > 0 else "✗ (boş)"
        print(f"{name:<22} {status:<8} {n:>6}  {ornek}")
        if n > 0:
            calisan += 1
        else:
            calismayanlar.append((name, url, "boş feed"))
    except Exception as e:
        print(f"{name:<22} {'✗ ERR':<8} {'—':>6}  {e}")
        calismayanlar.append((name, url, str(e)))

print(f"\n{'='*90}")
print(f"Çalışan: {calisan}/{len(FEEDS)}")
if calismayanlar:
    print(f"\nÇalışmayanlar:")
    for name, url, reason in calismayanlar:
        print(f"  • {name}: {reason}")
    print(f"\nÇalışmayan feed'leri settings.py'den kaldırabilirsin.")
