import sys
import datetime as dt
import numpy as np
import pandas as pd
import skops.io as sio


def cal_features_and_return_one_day(m01: pd.DataFrame,
                                    instrument: str, contract: str, contract_multiplier: int,
                                    pre_settle: float, pre_spot_close: float,
                                    sub_win_width: int = 30, tot_bar_num: int = 240,
                                    amount_scale: float = 1e4, ret_scale: int = 100) -> pd.DataFrame:
    # aggregate variables
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
    m01["datetime"] = m01["timestamp"].map(dt.datetime.fromtimestamp)
    m01["vwap"] = (m01["amount"] / m01["volume"] / contract_multiplier * amount_scale).fillna(method="ffill")
    m01["vwap_cum"] = (m01["amount"].cumsum() / m01["volume"].cumsum() / contract_multiplier * amount_scale).fillna(method="ffill")
    m01["m01_return"] = (m01["vwap"] / m01["vwap"].shift(1).fillna(pre_settle) - 1) * ret_scale

    # basic price
    prev_day_close = m01["preclose"].iloc[0]
    this_day_open = m01["daily_open"].iloc[0]
    this_day_end_vwap = m01["vwap"].iloc[-1]

    m05 = m01.set_index("datetime")[agg_vars].resample("5T").aggregate(agg_methods).dropna(axis=0, how="all", subset=dropna_cols)
    m10 = m01.set_index("datetime")[agg_vars].resample("10T").aggregate(agg_methods).dropna(axis=0, how="all", subset=dropna_cols)
    m15 = m01.set_index("datetime")[agg_vars].resample("15T").aggregate(agg_methods).dropna(axis=0, how="all", subset=dropna_cols)

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
        "alpha01": (prev_day_close / pre_settle - 1) * ret_scale,
        "alpha02": (this_day_open / prev_day_close - 1) * ret_scale,
        "alpha03": {}, "alpha04": {}, "alpha05": {}, "alpha06": {},  # prices return
        "alpha07": {}, "alpha08": {}, "alpha09": {},  # #top #diff #return
        "alpha10": {}, "alpha11": {}, "alpha12": {},  # corr(vwap, volume)
        "alpha13": {}, "alpha14": {}, "alpha15": {},  # corr(m01_return, volume)
        "alpha16": {}, "alpha17": {},
        "alpha18": {}, "alpha19": {},  # chart
        "alpha20": {},  # skewness
        "rtm": {},
    }

    sub_win_num = int(tot_bar_num / sub_win_width)
    for t in range(1, sub_win_num):
        bar_num_before_t = t * sub_win_width
        norm_scale = np.sqrt(bar_num_before_t)
        m01_before_t = m01.iloc[0:bar_num_before_t, :]
        next_vwap, ts = m01.at[bar_num_before_t, "vwap"], m01.at[bar_num_before_t, "timestamp"]

        sorted_vwap_and_ret_by_volume = m01_before_t[["vwap", "m01_return", "volume"]].sort_values(by="volume", ascending=False)
        skewness = m01_before_t["m01_return"].skew()
        top01_bars = int(0.1 * bar_num_before_t)
        top02_bars = int(0.2 * bar_num_before_t)
        top05_bars = int(0.5 * bar_num_before_t)
        corr_top_01 = sorted_vwap_and_ret_by_volume.head(top01_bars).corr(method="spearman")
        corr_top_02 = sorted_vwap_and_ret_by_volume.head(top02_bars).corr(method="spearman")
        corr_top_05 = sorted_vwap_and_ret_by_volume.head(top05_bars).corr(method="spearman")

        res["tid"][t], res["timestamp"][t] = "T{:02d}".format(t), ts

        # --- alphas
        res["alpha03"][t] = (m01_before_t["vwap"].iloc[-1] / this_day_open - 1) / norm_scale * ret_scale
        res["alpha04"][t] = (m01_before_t["vwap_cum"].iloc[-1] / this_day_open - 1) / norm_scale * ret_scale
        res["alpha05"][t] = (m01_before_t["daily_high"].iloc[-1] / this_day_open - 1) / norm_scale * ret_scale
        res["alpha06"][t] = (m01_before_t["daily_low"].iloc[-1] / this_day_open - 1) / norm_scale * ret_scale

        res["alpha07"][t] = sorted_vwap_and_ret_by_volume["m01_return"].head(top01_bars).mean()
        res["alpha08"][t] = sorted_vwap_and_ret_by_volume["m01_return"].head(top02_bars).tail(top02_bars - top01_bars).mean()
        res["alpha09"][t] = sorted_vwap_and_ret_by_volume["m01_return"].head(top05_bars).tail(top05_bars - top02_bars).mean()

        res["alpha10"][t] = corr_top_01.at["vwap", "volume"]
        res["alpha11"][t] = corr_top_02.at["vwap", "volume"]
        res["alpha12"][t] = corr_top_05.at["vwap", "volume"]

        res["alpha13"][t] = corr_top_01.at["m01_return", "volume"]
        res["alpha14"][t] = corr_top_02.at["m01_return", "volume"]
        res["alpha15"][t] = corr_top_05.at["m01_return", "volume"]

        res["alpha16"][t] = m01_before_t[["volume", "vwap"]].corr(method="spearman").at["vwap", "volume"]
        res["alpha17"][t] = m01_before_t[["volume", "m01_return"]].corr(method="spearman").at["m01_return", "volume"]

        if bar_num_before_t >= 15 * 3:
            res["alpha18"][t] = 1 if m15["low"][0] < m15["low"][1] < m15["low"][2] else 0
            res["alpha19"][t] = 1 if m15["high"][0] > m15["high"][1] > m15["high"][2] else 0
        elif bar_num_before_t >= 10 * 3:
            res["alpha18"][t] = 1 if m10["low"][0] < m10["low"][1] < m10["low"][2] else 0
            res["alpha19"][t] = 1 if m10["high"][0] > m10["high"][1] > m10["high"][2] else 0
        elif bar_num_before_t >= 5 * 3:
            res["alpha18"][t] = 1 if m05["low"][0] < m05["low"][1] < m05["low"][2] else 0
            res["alpha19"][t] = 1 if m05["high"][0] > m05["high"][1] > m05["high"][2] else 0
        else:
            res["alpha18"][t] = 0
            res["alpha19"][t] = 0

        res["alpha20"][t] = skewness

        # --- return to mature
        res["rtm"][t] = (this_day_end_vwap / next_vwap - 1) * ret_scale

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
