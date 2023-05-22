import sys
import datetime as dt
import numpy as np
import pandas as pd
import skops.io as sio


def cal_features_and_return(df: pd.DataFrame,
                            instrument: str, contract: str, contract_multiplier: int,
                            pre_settle: float, pre_spot_close: float,
                            sub_win_width: int = 30, tot_bar_num: int = 240,
                            amount_scale: float = 1e4, ret_scale: int = 100) -> pd.DataFrame:
    agg_vars = ["open", "high", "low", "close", "volume", "amount"]
    agg_methods = {
        "open": "first",
        "high": max,
        "low": min,
        "close": "last",
        "volume": np.sum,
        "amount": np.sum,
    }
    dropna_cols = ["open", "high", "low", "close"]

    # intermediary variables
    df["datetime"] = df["timestamp"].map(dt.datetime.fromtimestamp)
    df["vwap"] = (df["amount"] / df["volume"] / contract_multiplier * amount_scale).fillna(method="ffill")
    df["m01_return"] = (df["vwap"] / df["vwap"].shift(1).fillna(pre_settle) - 1) * ret_scale

    # basic price
    pre_close = df["preclose"].iloc[0]
    this_open = df["daily_open"].iloc[0]
    last_vwap = df["vwap"].iloc[-1]

    m05 = df.set_index("datetime")[agg_vars].resample("5T").aggregate(agg_methods).dropna(axis=0, how="all", subset=dropna_cols)
    m10 = df.set_index("datetime")[agg_vars].resample("10T").aggregate(agg_methods).dropna(axis=0, how="all", subset=dropna_cols)
    m15 = df.set_index("datetime")[agg_vars].resample("15T").aggregate(agg_methods).dropna(axis=0, how="all", subset=dropna_cols)

    for m_agg, m_agg_width in zip((m05, m10, m15), (5, 10, 15)):
        if len(m_agg) != tot_bar_num / m_agg_width:
            print("... data length is wrong! Length of M{:02d} is {} != {}".format(
                m_agg_width, len(m_agg), tot_bar_num / m_agg_width))
            print("... contract = {}".format(contract))
            print("... this program will terminate at once, please check again")
            sys.exit()
    res = {
        "instrument": instrument,
        "contract": contract,
        "tid": {}, "timestamp": {},
        "alpha00": (pre_settle / pre_spot_close - 1) * ret_scale,
        "alpha01": (pre_close / pre_settle - 1) * ret_scale,
        "alpha02": (this_open / pre_close - 1) * ret_scale,
        "alpha03": {},
        "alpha04": {}, "alpha05": {}, "alpha06": {},
        "alpha07": {}, "alpha08": {},
        "alpha09": {}, "alpha10": {}, "alpha11": {},
        "alpha12": {}, "alpha13": {}, "alpha14": {},
        "alpha15": {}, "alpha16": {},
        "alpha17": {}, "alpha18": {},
        "rtm": {},
    }

    sub_win_num = int(tot_bar_num / sub_win_width)
    for t in range(1, sub_win_num):
        bar_num_before_t = t * sub_win_width
        norm_scale = np.sqrt(bar_num_before_t)
        df_before_t = df.iloc[0:bar_num_before_t, :]
        next_vwap, ts = df.at[bar_num_before_t, "vwap"], df.at[bar_num_before_t, "timestamp"]

        sorted_return = df_before_t["m01_return"].sort_values(ascending=False)
        sorted_return_by_volume = df_before_t[["m01_return", "volume"]].sort_values(by="volume", ascending=False)

        res["tid"][t], res["timestamp"][t] = "T{:02d}".format(t), ts
        res["alpha03"][t] = (df_before_t["vwap"].iloc[-1] / this_open - 1) / norm_scale * ret_scale
        res["alpha04"][t] = sorted_return_by_volume.head(int(0.1 * bar_num_before_t)).mean()["m01_return"] * ret_scale
        res["alpha05"][t] = sorted_return_by_volume.head(int(0.2 * bar_num_before_t)).mean()["m01_return"] * ret_scale
        res["alpha06"][t] = sorted_return_by_volume.head(int(0.5 * bar_num_before_t)).mean()["m01_return"] * ret_scale
        res["alpha07"][t] = (df_before_t["daily_high"].iloc[-1] / this_open - 1) / norm_scale * ret_scale
        res["alpha08"][t] = (df_before_t["daily_low"].iloc[-1] / this_open - 1) / norm_scale * ret_scale
        res["alpha09"][t] = sorted_return.head(int(0.1 * bar_num_before_t)).mean() * ret_scale
        res["alpha10"][t] = sorted_return.head(int(0.2 * bar_num_before_t)).mean() * ret_scale
        res["alpha11"][t] = sorted_return.head(int(0.5 * bar_num_before_t)).mean() * ret_scale
        res["alpha12"][t] = sorted_return.tail(int(0.5 * bar_num_before_t)).mean() * ret_scale
        res["alpha13"][t] = sorted_return.tail(int(0.2 * bar_num_before_t)).mean() * ret_scale
        res["alpha14"][t] = sorted_return.tail(int(0.1 * bar_num_before_t)).mean() * ret_scale
        if bar_num_before_t >= 15 * 3:
            res["alpha15"][t] = 1 if m15["low"][0] < m15["low"][1] < m15["low"][2] else 0
            res["alpha16"][t] = 1 if m15["high"][0] > m15["high"][1] > m15["high"][2] else 0
        elif bar_num_before_t >= 10 * 3:
            res["alpha15"][t] = 1 if m10["low"][0] < m10["low"][1] < m10["low"][2] else 0
            res["alpha16"][t] = 1 if m10["high"][0] > m10["high"][1] > m10["high"][2] else 0
        elif bar_num_before_t >= 5 * 3:
            res["alpha15"][t] = 1 if m05["low"][0] < m05["low"][1] < m05["low"][2] else 0
            res["alpha16"][t] = 1 if m05["high"][0] > m05["high"][1] > m05["high"][2] else 0
        else:
            res["alpha15"][t] = 0
            res["alpha16"][t] = 0
        res["alpha17"][t] = df_before_t[["volume", "vwap"]].corr(method="spearman").at["vwap", "volume"]
        res["alpha18"][t] = df_before_t[["volume", "m01_return"]].corr(method="spearman").at["m01_return", "volume"]

        res["rtm"][t] = (last_vwap / next_vwap - 1) * ret_scale

    res_df = pd.DataFrame(res)
    return res_df


def save_to_sio_obj(t_sklearn_obj, t_path: str):
    obj = sio.dumps(t_sklearn_obj)
    with open(t_path, "wb+") as f:
        f.write(obj)
    return 0


def read_from_sio_obj(t_path: str):
    with open(t_path, "rb") as f:
        obj = f.read()
    return sio.loads(obj, trusted=True)
