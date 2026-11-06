"""1-min GOES XRS series of the flares of this study, read from data/input/xrs_windows.csv.gz."""
import pandas as pd
from paths import INP


class Cache:
    def __init__(self):
        w = pd.read_csv(INP / "xrs_windows.csv.gz", dtype={"sat": str})
        w["t"] = pd.to_datetime(w.t, utc=True, format="ISO8601")
        self.g = {k: v for k, v in w.groupby(["t0", "t1"], sort=False)}

    def window(self, t0, t1):
        v = self.g.get((t0.isoformat(), t1.isoformat()))
        if v is None:
            return {}
        d = v.set_index("t")[["short", "long"]].rename(columns={"short": "Flux_Short", "long": "Flux_Long"})
        return {(v.sat.iloc[0], v.src.iloc[0]): d}
