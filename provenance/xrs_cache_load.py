"""Read 1-min GOES XRS records from the local cache (sqlite pickles written by the project's own fetcher)."""
import pickle, sqlite3, zlib
from pathlib import Path
import numpy as np, pandas as pd, netCDF4
ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / "data/goes_cache/goes_xrs_series_cache.sqlite"
SEM = ROOT / "data/goes_cache/sem_xrs"

class Cache:
    def __init__(self):
        self.c = sqlite3.connect(DB)
        k = pd.read_sql("select cache_key,span_a,span_b from sunpy_series_blob where kind='goes_xrs_records'", self.c)
        k["a"] = pd.to_datetime(k.span_a, utc=True); k["b"] = pd.to_datetime(k.span_b, utc=True) + pd.Timedelta(days=1)
        k["sat"] = k.cache_key.str.extract(r"\|sat(\w+)\|")[0]
        k["src"] = k.cache_key.str.split("|").str[-1]
        self.k = k
    def window(self, t0, t1):
        """All cached 1-min records covering [t0,t1]; returns one frame per source key, best coverage first."""
        k = self.k[(self.k.a <= t1) & (self.k.b >= t0)]
        out = {}
        for r in k.itertuples():
            b = self.c.execute("select payload from sunpy_series_blob where cache_key=?", (r.cache_key,)).fetchone()[0]
            d = pd.DataFrame(pickle.loads(zlib.decompress(b)))
            if not len(d): continue
            d["t"] = pd.to_datetime(d.DateTime, utc=True)
            d = d[(d.t >= t0) & (d.t <= t1)]
            if len(d): out.setdefault((r.sat, r.src), []).append(d[["t", "Flux_Short", "Flux_Long"]])
        for f in sorted(SEM.glob("goes*/g*_xrs_1m_*.nc")):
            a, b = f.stem.split("_")[-2:]
            a = pd.Timestamp(a, tz="UTC"); b = pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1)
            if a > t1 or b < t0: continue
            with netCDF4.Dataset(f) as n:
                tt = pd.to_datetime(np.asarray(n["time_tag"][:], dtype="int64"), unit="ms", utc=True)
                m = (tt >= t0) & (tt <= t1)
                if not m.any(): continue
                vs, vl = ("xs", "xl") if "xs" in n.variables else ("A_AVG", "B_AVG") if "A_AVG" in n.variables else ("A_FLUX", "B_FLUX")
                n.set_auto_mask(False)
                d = pd.DataFrame({"t": tt[m], "Flux_Short": np.asarray(n[vs][:], float)[m], "Flux_Long": np.asarray(n[vl][:], float)[m]})
            out.setdefault((f.stem.split("_")[0][1:], "sem_nc"), []).append(d.dropna())
        res = {}
        for key, v in out.items():
            d = pd.concat(v).drop_duplicates("t").sort_values("t").set_index("t")
            d = d[(d.Flux_Long > 0)]
            res[key] = d
        return res
