import argparse
import os
import datetime as dt
import sys
import pandas as pd
from project_setup import calendar_path
from project_setup import research_features_and_return_dir
from falkreath import CManagerLibReader, CTable
from whiterun import CCalendarMonthly
from project_config import sqlite3_tables
from project_config import train_windows
from project_config import x_lbls, y_lbls

parser = argparse.ArgumentParser(description="A base model, using vanilla linear regression")
parser.add_argument("--instru", type=str, default=None, help="like IC.CFE")
parser.add_argument("--tid", type=str, default=None, help="must be in [t01, ..., t07]")
parser.add_argument("--bgn", type=str, help="begin date, format = [YYYYMMDD]", required=True)
parser.add_argument("--stp", type=str, default=None, help="stop date, format = [YYYYMMDD], can be skip, and program will use bgn only")
args = parser.parse_args()

instrument, tid = args.instru, args.tid
conds = {k: v for k, v in zip(("instrument", "tid"), (instrument, tid)) if v is not None}
print(conds)

bgn_date, stp_date = args.bgn, args.stp
if stp_date is None:
    stp_date = (dt.datetime.strptime(bgn_date, "%Y%m%d") + dt.timedelta(days=1)).strftime("%Y%m%d")

# --- load calendar
calendar = CCalendarMonthly(calendar_path)

# --- load lib writer
features_and_return_lib = CManagerLibReader(
    t_db_save_dir=research_features_and_return_dir,
    t_db_name="features_and_return.db"
)
features_and_return_tab = CTable(t_table_struct=sqlite3_tables["features_and_return"])
features_and_return_lib.set_default(features_and_return_tab.m_table_name)

# ---
iter_dates = calendar.get_iter_list(bgn_date, stp_date, True)
bgn_last_month = calendar.get_latest_month_from_trade_date(iter_dates[0])
end_last_month = calendar.get_latest_month_from_trade_date(iter_dates[-1])
stp_last_month = calendar.get_next_month(end_last_month, 1)
iter_months = calendar.get_iter_month(bgn_last_month, stp_last_month)

for train_end_month in iter_months:
    for trn_win in train_windows:
        train_bgn_month = calendar.get_next_month(train_end_month, -trn_win + 1)
        train_bgn_date = calendar.get_first_date_of_month(train_bgn_month)
        train_end_date = calendar.get_last_date_of_month(train_end_month)
        train_stp_date = calendar.get_next_date(train_end_date, 1)
        print(train_end_month, trn_win, train_bgn_date, train_stp_date)

        src_df = features_and_return_lib.read_by_conditions_and_time_window(
            t_conditions=conds, t_conditions_relation=0,
            t_bgn_date=train_bgn_date, t_stp_date=train_stp_date,
            t_value_columns=["trade_date", "contract", "tid"] + x_lbls + y_lbls
        )
        x_df = src_df[x_lbls]
        y_df = src_df[y_lbls]
        print(src_df)
        print(x_df.corr())
        print(x_df.abs().sum())
        print("=" * 120)

    break
    # print("... linear regression mode for {} is trained @ {}".format(train_end_month, dt.datetime.now()))
features_and_return_lib.close()
