"""Keşif istatistikleri, test sonuçlarının özeti ve grafikler.

Girdi : outputs/tahminler.csv, outputs/gunluk.csv   (02_model.py çıktısı)
        data/tuketim_saatlik.csv, data/sicaklik_saatlik.csv
        data/yuk_tahmin_plani.csv                   (varsa: EPİAŞ'ın resmi gün öncesi tahmini)
Çıktı : outputs/summary.json, outputs/*.png
"""
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("outputs")
ACCENT, GRAY, WARM = "#6d28d9", "#8a8799", "#d95926"
YONTEMLER = ["naif_7", "naif_2", "model", "model_hava"]
TIPLER = ["iş günü", "hafta sonu", "resmi tatil", "arife", "dini bayram"]
AYLAR = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e1ec", "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 10})

_spec = importlib.util.spec_from_file_location("model02", Path(__file__).with_name("02_model.py"))
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)


def y2(x: float, n: int = 2) -> float:
    return round(float(x), n)


def mape(tahmin: pd.Series, gercek: pd.Series) -> float:
    return float(np.mean(np.abs(tahmin / gercek - 1)) * 100)


def hata_ozeti(t: pd.DataFrame, yontem: str) -> dict:
    e = t[yontem] - t["y"]
    return {"mape": y2(mape(t[yontem], t["y"])), "mae": y2(e.abs().mean(), 0), "sapma": y2((t[yontem].sum() / t["y"].sum() - 1) * 100),
            "p95": y2(np.percentile(np.abs(e / t["y"]) * 100, 95)), "n": int(len(t))}


def kesif(df: pd.DataFrame, g: pd.DataFrame) -> dict:
    y = df["y"]
    S = {"veri": {"saat": int(len(y)), "bas": f"{y.index.min():%Y-%m-%d}", "bit": f"{y.index.max():%Y-%m-%d}",
                  "yillik_twh": {int(k.year): y2(v / 1e6, 1) for k, v in y.resample("YE").sum().items() if y[str(k.year)].size >= 8760},
                  "rekor": {"zaman": f"{y.idxmax():%Y-%m-%d %H:%M}", "mw": y2(y.max(), 0)},
                  "ortalama": y2(y.mean(), 0), "en_dusuk": {"zaman": f"{y.idxmin():%Y-%m-%d %H:%M}", "mw": y2(y.min(), 0)}}}

    isg = df.index.dayofweek < 5
    S["profil"] = {"is_gunu": [y2(v, 3) for v in (y[isg].groupby(y[isg].index.hour).mean() / y.mean())],
                   "hafta_sonu": [y2(v, 3) for v in (y[~isg].groupby(y[~isg].index.hour).mean() / y.mean())],
                   "gunler": [y2(v, 3) for v in (y.groupby(y.index.dayofweek).mean() / y.mean())]}

    # Tatil etkisi: günü, ±3 hafta içindeki aynı haftanın günlerinin (tatil olmayan) medyanıyla kıyasla
    normal = g[g["tip"].isin(["iş günü", "hafta sonu"])]["y"]
    ref = []
    for d in g.index:
        v = [normal.get(d + pd.Timedelta(days=7 * k)) for k in (-3, -2, -1, 1, 2, 3)]
        v = [x for x in v if x is not None and not np.isnan(x)]
        ref.append(np.median(v) if v else np.nan)
    g = g.assign(ref=ref)
    g["fark"] = (g["y"] / g["ref"] - 1) * 100
    S["tatil"] = {"tip": {t: {"gun": int((g["tip"] == t).sum()), "fark": y2(g.loc[g["tip"] == t, "fark"].mean(), 1)} for t in ("resmi tatil", "arife", "dini bayram")},
                  "ad": {a: {"gun": int(len(x)), "fark": y2(x["fark"].mean(), 1)} for a, x in g.dropna(subset=["tatil_adi"]).groupby("tatil_adi") if len(x) >= 5},
                  "bayram_gunu": {int(k): y2(v, 1) for k, v in g[g["bayram_gunu"] > 0].groupby("bayram_gunu")["fark"].mean().items()}}
    ay = y.resample("MS").mean()
    S["pandemi"] = {"nisan_2020": y2((ay["2020-04-01"] / ay["2019-04-01"] - 1) * 100, 1), "mayis_2020": y2((ay["2020-05-01"] / ay["2019-05-01"] - 1) * 100, 1)}

    # Sıcaklık eğrisi: tatil olmayan iş günleri, tüketim o yılın iş günü ortalamasına oranlanır
    w = g[(g["tip"] == "iş günü") & g["T"].notna()].copy()
    w["oran"] = w["y"] / w.groupby(w.index.year)["y"].transform("mean")
    kenar = [-10, 2, 6, 10, 14, 18, 22, 26, 40]
    etiket = ["<2", "2–6", "6–10", "10–14", "14–18", "18–22", "22–26", "26+"]
    w["dilim"] = pd.cut(w["T"], kenar, labels=etiket)
    S["sicaklik"] = {"dilim": etiket,
                     "hepsi": [y2((v - 1) * 100, 1) for v in w.groupby("dilim", observed=False)["oran"].mean()],
                     "gun": [int(v) for v in w.groupby("dilim", observed=False)["oran"].size()],
                     "ilk": [y2((v - 1) * 100, 1) for v in w[w.index.year <= 2019].groupby("dilim", observed=False)["oran"].mean()],
                     "son": [y2((v - 1) * 100, 1) for v in w[w.index.year >= 2023].groupby("dilim", observed=False)["oran"].mean()]}
    return S


def kalibre(t: pd.DataFrame, hedef: float = 0.8, pencere_ay: int = 12) -> pd.DataFrame:
    """Aralıkları, önceki ayların test hatalarına bakarak genişletir (conformalized quantile regression).

    Her ay için yalnızca o aydan önce gerçekleşmiş saatler kullanılır; ilk ayda geçmiş olmadığı için ham aralık kalır.
    """
    t = t.copy()
    skor = np.maximum(t["alt"] / t["y"] - 1, 1 - t["ust"] / t["y"])  # aralığın dışına ne kadar taşıldı (içerideyse negatif)
    t["alt_k"], t["ust_k"] = t["alt"], t["ust"]
    ay = t.index.to_period("M")
    for a in ay.unique()[1:]:
        gecmis = skor[(ay < a) & (ay >= a - pencere_ay)]
        q = float(np.quantile(gecmis, hedef))
        t.loc[ay == a, "alt_k"] = t.loc[ay == a, "alt"] * (1 - q)
        t.loc[ay == a, "ust_k"] = t.loc[ay == a, "ust"] * (1 + q)
    return t


def test_ozeti(t: pd.DataFrame) -> dict:
    R = {"donem": [f"{t.index.min():%Y-%m-%d}", f"{t.index.max():%Y-%m-%d}"], "saat": int(len(t)), "gun": int(t.index.normalize().nunique())}
    R["genel"] = {m: hata_ozeti(t, m) for m in YONTEMLER}
    R["yil"] = {int(y): {m: y2(mape(x[m], x["y"])) for m in YONTEMLER} for y, x in t.groupby(t.index.year)}
    R["tip"] = {tip: {"gun": int(x.index.normalize().nunique()), **{m: y2(mape(x[m], x["y"])) for m in YONTEMLER}} for tip in TIPLER for x in [t[t["tip"] == tip]] if len(x)}
    R["saatlik"] = {m: [y2(mape(x[m], x["y"])) for _, x in t.groupby(t.index.hour)] for m in ("naif_7", "model", "model_hava")}
    R["ay"] = {m: [y2(mape(x[m], x["y"])) for _, x in t.groupby(t.index.month)] for m in ("naif_7", "model", "model_hava")}
    # Günün en yüksek saati: şebeke için en kritik değer
    gun = t.groupby(t.index.normalize())
    tepe = pd.DataFrame({"y": gun["y"].max(), **{m: gun[m].max() for m in YONTEMLER}})
    R["tepe"] = {m: y2(mape(tepe[m], tepe["y"])) for m in YONTEMLER}
    # Günlük toplam
    top = gun[["y"] + YONTEMLER].sum()
    R["gunluk_toplam"] = {m: y2(mape(top[m], top["y"])) for m in YONTEMLER}
    # En kötü günler
    gm = gun.apply(lambda x: mape(x["model_hava"], x["y"]))
    kotu = gm.sort_values(ascending=False).head(12)
    R["en_kotu"] = [{"gun": f"{d:%Y-%m-%d}", "mape": y2(v, 1), "tip": t.loc[t.index.normalize() == d, "tip"].iat[0],
                     "sapma": y2((top.loc[d, "model_hava"] / top.loc[d, "y"] - 1) * 100, 1)} for d, v in kotu.items()]
    R["gun_dagilimi"] = {"medyan": y2(gm.median()), "p90": y2(gm.quantile(0.9)), "iki_alti": y2((gm < 2).mean() * 100, 1), "bes_ustu": y2((gm > 5).mean() * 100, 1)}
    # Tahmin aralığı (%10–%90)
    ic = (t["y"] >= t["alt"]) & (t["y"] <= t["ust"])
    R["aralik"] = {"kapsama": y2(ic.mean() * 100, 1), "alt_disi": y2((t["y"] < t["alt"]).mean() * 100, 1), "ust_disi": y2((t["y"] > t["ust"]).mean() * 100, 1),
                   "genislik": y2(((t["ust"] - t["alt"]) / t["y"]).mean() * 100, 1),
                   "tip": {tip: y2(ic[t["tip"] == tip].mean() * 100, 1) for tip in TIPLER if (t["tip"] == tip).any()}}
    k = t[t.index >= t.index.min() + pd.offsets.MonthBegin(1)]  # kalibrasyonun geçmişi olan aylar
    ick = (k["y"] >= k["alt_k"]) & (k["y"] <= k["ust_k"])
    R["aralik_kalibre"] = {"bas": f"{k.index.min():%Y-%m-%d}", "kapsama": y2(ick.mean() * 100, 1), "genislik": y2(((k["ust_k"] - k["alt_k"]) / k["y"]).mean() * 100, 1),
                           "ham_ayni_donem": y2(((k["y"] >= k["alt"]) & (k["y"] <= k["ust"])).mean() * 100, 1),
                           "ham_tip": {tip: y2(((k["y"] >= k["alt"]) & (k["y"] <= k["ust"]))[k["tip"] == tip].mean() * 100, 1) for tip in TIPLER if (k["tip"] == tip).any()},
                           "yil": {int(y): y2(v.mean() * 100, 1) for y, v in ick.groupby(k.index.year)},
                           "tip": {tip: y2(ick[k["tip"] == tip].mean() * 100, 1) for tip in TIPLER if (k["tip"] == tip).any()}}
    return R


def hava_duyarliligi(df: pd.DataFrame) -> dict:
    """Hava tahmini şaşarsa ne olur? 2025 öncesiyle eğitilen model, sıcaklığa gün bazında hata eklenerek sınanır."""
    X = M.ozellikler(df)
    hepsi = [c for c in X.columns if c not in M.DISARIDA]
    ok = X[hepsi + ["hedef"]].notna().all(axis=1)
    egitim, test = ok & (X.index < "2025-01-01"), ok & (X.index >= "2025-01-01")
    m = M.model().fit(X.loc[egitim, hepsi], X.loc[egitim, "hedef"])
    y = df.loc[test, "y"]
    havasiz = [c for c in hepsi if c not in M.HAVA]
    m0 = M.model().fit(X.loc[egitim, havasiz], X.loc[egitim, "hedef"])
    sicakliksiz = mape(X.loc[test, "ref"] * np.exp(m0.predict(X.loc[test, havasiz])), y)
    rng = np.random.default_rng(M.SEED)
    gunler = X.index[test].normalize()
    out = {}
    for sigma in (0, 1, 2, 3):
        hatalar = []
        for _ in range(5 if sigma else 1):
            Xt = X.loc[test, hepsi].copy()
            gurultu = pd.Series(rng.normal(0, sigma, gunler.nunique()), index=gunler.unique()).reindex(gunler).to_numpy()
            for c in ["T", "T_fark", "T_gun", "T_gun_fark"] + [f"T_{s}" for s in M.SEHIRLER]:
                Xt[c] = Xt[c] + gurultu
            hatalar.append(mape(X.loc[test, "ref"] * np.exp(m.predict(Xt)), y))
        out[str(sigma)] = y2(np.mean(hatalar))
    return {"donem": [f"{y.index.min():%Y-%m-%d}", f"{y.index.max():%Y-%m-%d}"], "mape": out, "sicakliksiz": y2(sicakliksiz)}


def resmi_plan(t: pd.DataFrame) -> dict | None:
    """EPİAŞ'ta yayımlanan Yük Tahmin Planı (bir sonraki gün için resmi saatlik tahmin) ile aynı saatlerde kıyas."""
    yol = Path("data/yuk_tahmin_plani.csv")
    if not yol.exists():
        return None
    p = pd.read_csv(yol)
    deger = [c for c in p.columns if c not in ("date", "time")][0]
    p["zaman"] = pd.to_datetime(p["date"].str.slice(0, 19))
    plan = p.drop_duplicates("zaman").set_index("zaman")[deger]
    x = t.join(plan.rename("plan"), how="inner").dropna(subset=["plan"])
    x = x[x["plan"] > 0]
    ad = ("plan", "model", "model_hava", "naif_7")
    ape = {m: (x[m] / x["y"] - 1).abs() * 100 for m in ad}
    R = {"saat": int(len(x)), "eksik_saat": int(len(t) - len(x)), "genel": {m: hata_ozeti(x, m) for m in ad},
         "medyan": {m: y2(ape[m].median()) for m in ad},
         "yil": {int(y): {m: y2(mape(g[m], g["y"])) for m in ad[:3]} for y, g in x.groupby(x.index.year)},
         "tip": {tip: {m: y2(mape(g[m], g["y"])) for m in ad[:3]} for tip in TIPLER for g in [x[x["tip"] == tip]] if len(g)}}
    # Planda bariz hatalı görünen günler (günlük hata %15'in üstünde) çıkarılınca
    gunluk = ape["plan"].groupby(x.index.normalize()).mean()
    bozuk = gunluk[gunluk > 15].index
    temiz = x[~x.index.normalize().isin(bozuk)]
    R["plan_bozuk_gun"] = [{"gun": f"{d:%Y-%m-%d}", "mape": y2(gunluk[d], 1)} for d in gunluk[bozuk].sort_values(ascending=False).index]
    R["bozuk_haric"] = {m: y2(mape(temiz[m], temiz["y"])) for m in ad[:3]}
    # Gün bazında kim daha iyi?
    gm = pd.DataFrame({m: ape[m].groupby(x.index.normalize()).mean() for m in ad[:3]})
    R["gun_kazanan"] = {"model": y2((gm["model"] < gm["plan"]).mean() * 100, 1), "model_hava": y2((gm["model_hava"] < gm["plan"]).mean() * 100, 1), "gun": int(len(gm))}
    # İki tahminin ortalaması
    R["karma"] = {"model": y2(mape((x["plan"] + x["model"]) / 2, x["y"])), "model_hava": y2(mape((x["plan"] + x["model_hava"]) / 2, x["y"]))}
    return R


def grafikler(S: dict, t: pd.DataFrame, g: pd.DataFrame) -> None:
    # 1) Günlük profil
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(range(24), S["profil"]["is_gunu"], color=ACCENT, linewidth=2, label="İş günü")
    ax.plot(range(24), S["profil"]["hafta_sonu"], color=GRAY, linewidth=2, label="Hafta sonu")
    ax.set(xlabel="Saat", ylabel="Ortalamaya oran", xticks=range(0, 24, 3), title="Günün saatlerine göre tüketim")
    ax.legend(frameon=False)
    plt.tight_layout(); fig.savefig(OUT / "1_profil.png"); plt.close(fig)

    # 2) Sıcaklık eğrisi
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(len(S["sicaklik"]["dilim"]))
    ax.plot(x, S["sicaklik"]["ilk"], marker="o", color=GRAY, linewidth=2, label="2017–2019")
    ax.plot(x, S["sicaklik"]["son"], marker="o", color=ACCENT, linewidth=2, label="2023–2026")
    ax.axhline(0, color="#c9c5d6", linewidth=1)
    ax.set(xticks=x, xticklabels=S["sicaklik"]["dilim"], xlabel="Günlük ortalama sıcaklık (°C, beş şehir)", ylabel="Yıl ortalamasından fark (%)",
           title="Sıcaklık ve tüketim (tatil olmayan iş günleri)")
    ax.legend(frameon=False)
    plt.tight_layout(); fig.savefig(OUT / "2_sicaklik.png"); plt.close(fig)

    # 3) Bir bayram haftası: gerçekleşen, model, geçen hafta
    for bas, bit, ad in (("2026-05-22", "2026-06-04", "3_bayram_haftasi.png"),):
        x = t.loc[bas:bit]
        if len(x) == 0:
            continue
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(x.index, x["y"] / 1000, color="#2a2833", linewidth=1.6, label="Gerçekleşen")
        ax.plot(x.index, x["naif_7"] / 1000, color=GRAY, linewidth=1.2, linestyle="--", label="Geçen haftanın aynı saati")
        ax.plot(x.index, x["model_hava"] / 1000, color=ACCENT, linewidth=1.6, label="Model")
        ax.fill_between(x.index, x["alt_k"] / 1000, x["ust_k"] / 1000, color=ACCENT, alpha=0.12, linewidth=0)
        ax.set(ylabel="GW", title="Kurban Bayramı 2026: bayram haftasında tahmin")
        ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d.%m"))
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        plt.tight_layout(); fig.savefig(OUT / ad); plt.close(fig)

    # 4) Gün tipine göre hata
    R = S["test"]
    fig, ax = plt.subplots(figsize=(7, 3))
    tipler = [k for k in TIPLER if k in R["tip"]]
    yy = np.arange(len(tipler))
    ax.barh(yy - 0.2, [R["tip"][k]["naif_7"] for k in tipler], height=0.36, color=GRAY, label="Geçen haftanın aynı saati")
    ax.barh(yy + 0.2, [R["tip"][k]["model_hava"] for k in tipler], height=0.36, color=ACCENT, label="Model")
    for i, k in enumerate(tipler):
        ax.text(R["tip"][k]["naif_7"] + 0.3, i - 0.2, f"%{R['tip'][k]['naif_7']:.1f}", va="center", fontsize=8)
        ax.text(R["tip"][k]["model_hava"] + 0.3, i + 0.2, f"%{R['tip'][k]['model_hava']:.1f}", va="center", fontsize=8)
    ax.set(yticks=yy, yticklabels=[k.capitalize() for k in tipler], xlabel="Ortalama mutlak yüzde hata (MAPE)", title="Gün tipine göre tahmin hatası")
    ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.legend(frameon=False)
    plt.tight_layout(); fig.savefig(OUT / "4_gun_tipi.png"); plt.close(fig)

    # 5) Aylara göre hata
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(range(12), R["ay"]["model"], marker="o", color=GRAY, linewidth=2, label="Model, sıcaklıksız")
    ax.plot(range(12), R["ay"]["model_hava"], marker="o", color=ACCENT, linewidth=2, label="Model, sıcaklıkla")
    ax.set(xticks=range(12), xticklabels=AYLAR, ylabel="MAPE (%)", title="Sıcaklık bilgisi hangi aylarda işe yarıyor?", ylim=(0, None))
    ax.legend(frameon=False)
    plt.tight_layout(); fig.savefig(OUT / "5_aylar.png"); plt.close(fig)


def main() -> None:
    df = M.veri_oku()
    t = kalibre(pd.read_csv(OUT / "tahminler.csv", parse_dates=["zaman"]).set_index("zaman"))
    g = pd.read_csv(OUT / "gunluk.csv", parse_dates=["gun"]).set_index("gun")
    S = kesif(df, g)
    S["test"] = test_ozeti(t)
    S["hava_duyarliligi"] = hava_duyarliligi(df)
    plan = resmi_plan(t)
    if plan:
        S["resmi_plan"] = plan
    (OUT / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=2), encoding="utf-8")
    grafikler(S, t, g)
    print(json.dumps({k: S[k] for k in S if k not in ("profil",)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
