"""Türkiye saatlik elektrik tüketimini (EPİAŞ) ve beş şehrin saatlik sıcaklığını (Open-Meteo) indirir.

Çıktı:
    data/tuketim_saatlik.csv     EPİAŞ Şeffaflık Platformu, Gerçek Zamanlı Tüketim (MWh)
    data/yuk_tahmin_plani.csv    EPİAŞ Şeffaflık Platformu, Yük Tahmin Planı (resmi gün öncesi tahmin, MWh)
    data/sicaklik_saatlik.csv    Open-Meteo arşivi, şehir başına 2 metre sıcaklığı (°C)

Kurulum:
    pip install eptr2 pandas requests

Kullanım:
    python 01_veri_cek.py                    # 2017'den bugüne
    python 01_veri_cek.py --baslangic 2020-01-01

EPİAŞ kullanıcı adı ve şifresi çalışırken sorulur; hiçbir yere yazılmaz.
Yarıda kesilirse tekrar çalıştır: indirilmiş ayları atlar.
"""
import argparse
import getpass
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

# Nüfusu büyük ve iklimi birbirinden farklı beş şehir: (enlem, boylam)
SEHIRLER = {
    "istanbul": (41.01, 28.98),
    "ankara": (39.93, 32.86),
    "izmir": (38.42, 27.14),
    "adana": (37.00, 35.32),
    "erzurum": (39.90, 41.27),
}


def ay_araliklari(bas: date, bit: date):
    """[bas, bit] aralığını ay ay böler: API tek seferde uzun aralık vermeyebilir."""
    a = bas
    while a <= bit:
        sonraki = (a.replace(day=1) + timedelta(days=32)).replace(day=1)
        yield a, min(sonraki - timedelta(days=1), bit)
        a = sonraki


# EPİAŞ'tan indirilen seriler: (eptr2 çağrı adı, parça klasörü, çıktı dosyası)
SERILER = {
    "tuketim": ("rt-cons", "_parcalar", "tuketim_saatlik.csv"),            # Gerçek Zamanlı Tüketim
    "plan": ("load-plan", "_parcalar_plan", "yuk_tahmin_plani.csv"),       # Yük Tahmin Planı: resmi gün öncesi tahmin
}


def giris():
    """Kullanıcı adı ve şifre burada sorulur; hiçbir yere yazılmaz."""
    from eptr2 import EPTR2

    print("EPİAŞ Şeffaflık Platformu girişi (bilgiler kaydedilmez)")
    for deneme in (1, 2, 3):
        kullanici = input("  Kullanıcı adı (e-posta): ").strip()
        # Windows terminalinde Ctrl+V burada çalışmaz: şifreyi elle yaz ya da sağ tıkla yapıştır.
        sifre = getpass.getpass("  Şifre (yazarken görünmez): ").strip()
        try:
            return EPTR2(username=kullanici, password=sifre, use_dotenv=False, recycle_tgt=False)
        except Exception as e:
            if "401" not in str(e):
                raise
            print(f"  Giriş reddedildi ({len(sifre)} karakterlik şifre gönderildi). Aynı bilgilerle "
                  "seffaflik.epias.com.tr sitesine girebildiğini kontrol et; hesabı e-postadan onaylamış olman gerekir.")
    raise SystemExit("Giriş yapılamadı.")


def epias_indir(ad: str, bas: date, bit: date, klasor: Path, oturum: list) -> pd.DataFrame:
    cagri, parca_adi, cikti = SERILER[ad]
    parca_klasor = klasor / parca_adi
    parca_klasor.mkdir(parents=True, exist_ok=True)
    eksik = [(a, b) for a, b in ay_araliklari(bas, bit) if not (parca_klasor / f"{a:%Y-%m}.csv").exists() or b == bit]

    for a, b in eksik:
        if not oturum:
            oturum.append(giris())
        for deneme in (1, 2, 3):
            try:
                df = oturum[0].call(cagri, start_date=f"{a:%Y-%m-%d}", end_date=f"{b:%Y-%m-%d}")
                break
            except Exception as e:
                if deneme == 3:
                    raise SystemExit(f"{ad} {a:%Y-%m} indirilemedi: {e}\nTekrar çalıştırınca kaldığı yerden devam eder.")
                time.sleep(5)
        if df is None or len(df) == 0:
            print(f"  {a:%Y-%m}: veri yok")
            continue
        df.to_csv(parca_klasor / f"{a:%Y-%m}.csv", index=False)
        print(f"  {a:%Y-%m}: {len(df)} satır")
        time.sleep(0.5)  # sunucuyu yormamak için

    parcalar = sorted(parca_klasor.glob("*.csv"))
    if not parcalar:
        raise SystemExit(f"{ad}: hiç veri inmedi.")
    hepsi = pd.concat([pd.read_csv(p) for p in parcalar], ignore_index=True).drop_duplicates()
    hepsi.to_csv(klasor / cikti, index=False, encoding="utf-8-sig")
    return hepsi


def sicaklik_indir(bas: date, bit: date, klasor: Path) -> pd.DataFrame | None:
    """Open-Meteo arşivi yaklaşık bir hafta geriden gelir; anahtar istemez."""
    bit = min(bit, date.today() - timedelta(days=7))
    seriler = {}
    for sehir, (enlem, boylam) in SEHIRLER.items():
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=120, params={
            "latitude": enlem, "longitude": boylam, "start_date": f"{bas:%Y-%m-%d}", "end_date": f"{bit:%Y-%m-%d}",
            "hourly": "temperature_2m", "timezone": "Europe/Istanbul"})
        r.raise_for_status()
        saatlik = r.json()["hourly"]
        seriler[sehir] = pd.Series(saatlik["temperature_2m"], index=saatlik["time"])
        print(f"  {sehir}: {len(saatlik['time'])} saat")
        time.sleep(1)
    df = pd.DataFrame(seriler).rename_axis("zaman").reset_index()
    df.to_csv(klasor / "sicaklik_saatlik.csv", index=False, encoding="utf-8-sig")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baslangic", default="2017-01-01")
    ap.add_argument("--bitis", default=f"{date.today() - timedelta(days=1):%Y-%m-%d}")
    ap.add_argument("--klasor", default="data")
    args = ap.parse_args()
    bas, bit = date.fromisoformat(args.baslangic), date.fromisoformat(args.bitis)
    klasor = Path(args.klasor)
    klasor.mkdir(parents=True, exist_ok=True)

    oturum = []  # ilk ihtiyaçta giriş yapılır, iki seri aynı oturumu kullanır
    for ad, baslik in (("tuketim", "Gerçek zamanlı tüketim"), ("plan", "Yük tahmin planı")):
        print(f"\n{baslik}: {bas} – {bit}")
        t = epias_indir(ad, bas, bit, klasor, oturum)
        print(f"{len(t)} satır -> {(klasor / SERILER[ad][2]).resolve()}")
        print(t.head(2).to_string(index=False))
        print(t.tail(2).to_string(index=False))

    print("\nSıcaklık (Open-Meteo)")
    try:
        sicaklik_indir(bas, bit, klasor)
        print(f"-> {(klasor / 'sicaklik_saatlik.csv').resolve()}")
    except Exception as e:  # sıcaklık olmadan da analiz yapılabilir
        print(f"  sıcaklık indirilemedi ({e}); tüketim dosyası tamam, onu gönder.")


if __name__ == "__main__":
    main()
