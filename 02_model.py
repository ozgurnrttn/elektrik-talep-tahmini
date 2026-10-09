"""Gün öncesi saatlik elektrik tüketimi tahmini: özellikler, modeller ve geriye dönük test.

Girdi : data/tuketim_saatlik.csv, data/sicaklik_saatlik.csv   (01_veri_cek.py çıktısı)
Çıktı : outputs/tahminler.csv    test dönemindeki her saat için gerçekleşen değer ve her yöntemin tahmini
        outputs/gunluk.csv       gün bazında tüketim, sıcaklık ve gün tipi (keşif grafikleri için)

Kural: D gününün 24 saati, en geç D-2 günü sonuna kadar gerçekleşen tüketimle tahmin edilir.
Model her ayın başında, o güne kadar bilinen veriyle yeniden eğitilir.

Kurulum: pip install pandas numpy scikit-learn holidays
"""
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

SEED = 42
TEST_BAS = "2022-01-01"
OUT = Path("outputs")
SEHIRLER = ["istanbul", "ankara", "izmir", "adana", "erzurum"]
DINI = ("Ramazan Bayramı", "Kurban Bayramı")


def veri_oku() -> pd.DataFrame:
    t = pd.read_csv("data/tuketim_saatlik.csv")
    t["zaman"] = pd.to_datetime(t["date"].str.slice(0, 19))  # Türkiye 2016'dan beri sabit UTC+3: yaz saati yok
    y = t.set_index("zaman")["consumption"].sort_index()
    tam = pd.date_range(y.index.min(), y.index.max(), freq="h")
    assert y.index.equals(tam), "saatlik seride boşluk ya da tekrar var"
    s = pd.read_csv("data/sicaklik_saatlik.csv")
    s["zaman"] = pd.to_datetime(s["zaman"])
    s = s.set_index("zaman")[SEHIRLER]
    df = pd.DataFrame({"y": y}).join(s)
    df["T"] = df[SEHIRLER].mean(axis=1)
    return df


def takvim(gunler: pd.DatetimeIndex) -> pd.DataFrame:
    """Gün bazında tatil bilgisi. Dini bayram tarihleri `holidays` paketinden gelir."""
    tr = holidays.Turkey(years=range(gunler[0].year - 1, gunler[-1].year + 2))
    ad = pd.Series({pd.Timestamp(d): n for d, n in tr.items()})
    g = pd.DataFrame(index=gunler)
    g["tatil_adi"] = ad.reindex(gunler)
    dini = g["tatil_adi"].fillna("").str.contains("|".join(DINI))
    g["dini"] = dini.astype(int)
    g["resmi"] = (g["tatil_adi"].notna() & ~dini).astype(int)
    # bayramın kaçıncı günü (1..4), arife (bayramdan önceki gün) ve bayram ertesi
    blok = (dini != dini.shift()).cumsum()
    g["bayram_gunu"] = np.where(dini, dini.groupby(blok).cumsum(), 0)
    g["arife"] = (dini.shift(-1, fill_value=False) & ~dini).astype(int)
    g["bayram_ertesi"] = (dini.shift(1, fill_value=False) & ~dini).astype(int)
    # Ramazan ayı: Ramazan Bayramı'nın ilk gününden önceki 30 gün (yaklaşık)
    ilk = g.index[(g["bayram_gunu"] == 1) & g["tatil_adi"].fillna("").str.contains("Ramazan")]
    g["ramazan"] = 0
    for b in ilk:
        g.loc[b - pd.Timedelta(days=30): b - pd.Timedelta(days=1), "ramazan"] = 1
    # en yakın dini bayrama gün farkı (−7..+7 dışı kırpılır): bayram haftasının etkisi için
    bayram_gunleri = g.index[dini]
    fark = np.array([(bayram_gunleri - d).days[np.abs((bayram_gunleri - d).days).argmin()] for d in gunler])
    g["bayrama_gun"] = np.clip(fark, -7, 7)
    # köprü: iki yanı da boş (tatil ya da hafta sonu) olan ve en az bir yanı tatil olan iş günü
    tatil = ((g["dini"] + g["resmi"]) > 0).to_numpy()
    hs = np.asarray(gunler.dayofweek >= 5)
    bos = tatil | hs
    once, sonra = np.roll(bos, 1), np.roll(bos, -1)
    g["kopru"] = (~bos & once & sonra & (np.roll(tatil, 1) | np.roll(tatil, -1))).astype(int)
    g["tip"] = np.select([g["dini"] == 1, g["arife"] == 1, g["resmi"] == 1, hs], ["dini bayram", "arife", "resmi tatil", "hafta sonu"], "iş günü")
    return g


def ozellikler(df: pd.DataFrame) -> pd.DataFrame:
    """Her saat için, D-2 günü sonuna kadar bilinenlerden üretilen özellikler."""
    y = df["y"]
    X = pd.DataFrame(index=df.index)
    X["saat"] = df.index.hour
    X["hafta_gunu"] = df.index.dayofweek
    gun_no = df.index.dayofyear
    X["yil_sin"], X["yil_cos"] = np.sin(2 * np.pi * gun_no / 365.25), np.cos(2 * np.pi * gun_no / 365.25)

    # Referans: aynı saatin D-2 … D-8 ortalaması (haftanın her günü bir kez). Hedef bu seviyeye orandır;
    # böylece ağaç modeli yıllar içinde artan tüketim seviyesini öğrenmek zorunda kalmaz.
    gecmis = pd.concat({k: y.shift(24 * k) for k in range(2, 9)}, axis=1)
    X["ref"] = gecmis.mean(axis=1, skipna=False)
    for k in (2, 3, 7, 14):
        X[f"oran_g{k}"] = y.shift(24 * k) / X["ref"]
    gunluk = y.resample("D").mean()
    g2 = gunluk.shift(2).reindex(df.index.normalize()).to_numpy()
    hafta = gunluk.rolling(7).mean().shift(2).reindex(df.index.normalize()).to_numpy()
    onceki_hafta = gunluk.rolling(7).mean().shift(9).reindex(df.index.normalize()).to_numpy()
    X["g2_oran"] = g2 / hafta
    X["hafta_egilim"] = hafta / onceki_hafta

    g = takvim(pd.date_range(df.index.min().normalize(), df.index.max().normalize(), freq="D"))
    gun = df.index.normalize()
    for c in ("dini", "resmi", "bayram_gunu", "arife", "bayram_ertesi", "ramazan", "bayrama_gun", "kopru"):
        X[c] = g[c].reindex(gun).to_numpy()
    tatil = (g["dini"] + g["resmi"] + g["arife"] > 0).astype(int)
    # referans günleri tatilse referans düşük kalır: model bunu bilmeli
    X["ref_tatil"] = sum(tatil.shift(k) for k in range(2, 9)).reindex(gun).to_numpy()
    X["g2_tatil"] = tatil.shift(2).reindex(gun).to_numpy()
    X["g7_tatil"] = tatil.shift(7).reindex(gun).to_numpy()
    X["yarin_tatil"] = tatil.shift(-1).reindex(gun).to_numpy()
    X["dun_tatil"] = tatil.shift(1).reindex(gun).to_numpy()

    # Sıcaklık: tahmin gününün sıcaklığı gerçekleşen değerdir (kusursuz hava tahmini varsayımı).
    T = df["T"]
    T_ref = pd.concat({k: T.shift(24 * k) for k in range(2, 9)}, axis=1).mean(axis=1, skipna=False)
    X["T"] = T
    X["T_fark"] = T - T_ref
    Tg = T.resample("D").mean()
    X["T_gun"] = Tg.reindex(gun).to_numpy()
    X["T_gun_fark"] = (Tg - Tg.rolling(7).mean().shift(2)).reindex(gun).to_numpy()
    X["T_dun"] = Tg.shift(1).reindex(gun).to_numpy()
    for s in SEHIRLER:
        X[f"T_{s}"] = df[s]
    X["hedef"] = np.log(y / X["ref"])
    X["tip"] = g["tip"].reindex(gun).to_numpy()
    return X


HAVA = ["T", "T_fark", "T_gun", "T_gun_fark", "T_dun"] + [f"T_{s}" for s in SEHIRLER]
DISARIDA = ["hedef", "tip", "ref"]


def model(kayip: str = "squared_error", q: float | None = None) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss=kayip, quantile=q, max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                         min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)


def geriye_donuk_test(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    hepsi = [c for c in X.columns if c not in DISARIDA]
    havasiz = [c for c in hepsi if c not in HAVA]
    gecerli = X[hepsi + ["hedef"]].notna().all(axis=1)
    aylar = pd.date_range(TEST_BAS, X.index[gecerli].max(), freq="MS")
    parcalar = []
    for ay in aylar:
        # ay başında bilinen son gün: iki gün öncesi
        egitim = gecerli & (X.index < ay - pd.Timedelta(days=1))
        test = gecerli & (X.index >= ay) & (X.index < ay + pd.offsets.MonthBegin(1))
        if not test.any():
            continue
        p = pd.DataFrame({"y": df.loc[test, "y"], "ref": X.loc[test, "ref"]})
        for ad, sutunlar in (("model", havasiz), ("model_hava", hepsi)):
            m = model().fit(X.loc[egitim, sutunlar], X.loc[egitim, "hedef"])
            p[ad] = p["ref"] * np.exp(m.predict(X.loc[test, sutunlar]))
        for ad, q in (("alt", 0.1), ("ust", 0.9)):
            m = model("quantile", q).fit(X.loc[egitim, hepsi], X.loc[egitim, "hedef"])
            p[ad] = p["ref"] * np.exp(m.predict(X.loc[test, hepsi]))
        parcalar.append(p)
        print(f"{ay:%Y-%m}  eğitim {int(egitim.sum()):>6}  MAPE model {np.mean(np.abs(p.model / p.y - 1)) * 100:.2f}  hava {np.mean(np.abs(p.model_hava / p.y - 1)) * 100:.2f}", flush=True)
    t = pd.concat(parcalar)
    t["naif_7"] = df["y"].shift(168).reindex(t.index)   # geçen haftanın aynı saati
    t["naif_2"] = df["y"].shift(48).reindex(t.index)    # iki gün öncesinin aynı saati
    t["tip"] = X["tip"].reindex(t.index)
    t["T"] = df["T"].reindex(t.index)
    return t


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = veri_oku()
    X = ozellikler(df)
    t = geriye_donuk_test(df, X)
    t.rename_axis("zaman").round(2).to_csv(OUT / "tahminler.csv")
    g = takvim(pd.date_range(df.index.min().normalize(), df.index.max().normalize(), freq="D"))
    gunluk = df[["y", "T"]].resample("D").mean().join(g[["tatil_adi", "tip", "bayram_gunu", "ramazan"]])
    gunluk.rename_axis("gun").round(2).to_csv(OUT / "gunluk.csv")
    for ad in ("naif_7", "naif_2", "ref", "model", "model_hava"):
        print(f"{ad:<11} MAPE %{np.mean(np.abs(t[ad] / t.y - 1)) * 100:.2f}")
    print("aralık kapsaması %", round(((t.y >= t.alt) & (t.y <= t.ust)).mean() * 100, 1))


if __name__ == "__main__":
    main()
