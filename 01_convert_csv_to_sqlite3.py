import argparse
import os
import datetime as dt

import pandas as pd

from project_setup import calendar_path
from project_setup import research_features_and_return_dir
from falkreath import CManagerLibWriterByDate, CTable
from whiterun import CCalendar
from project_config import equity_indexes
from project_config import sqlite3_tables

parser = argparse.ArgumentParser(description="A python version interface to view TSDB.")
parser.add_argument("--mode", type=str, help="must be one of ['o', 'overwrite', 'a', 'append']", required=True)
parser.add_argument("--bgn", type=str, help="begin date, format = [YYYYMMDD]", required=True)
parser.add_argument("--stp", type=str, default=None, help="stop date, format = [YYYYMMDD], can be skip, and program will use bgn only", required=False)
args = parser.parse_args()
print(args)

run_mode, bgn_date, stp_date = args.mode, args.bgn, args.stp
if stp_date is None:
    stp_date = (dt.datetime.strptime(bgn_date, "%Y%m%d") + dt.timedelta(days=1)).strftime("%Y%m%d")

# --- load calendar
calendar = CCalendar(calendar_path)

# --- load lib writer
features_and_return_lib = CManagerLibWriterByDate(
    t_db_save_dir=research_features_and_return_dir,
    t_db_name="features_and_return.db"
)
features_and_return_lib.initialize_table(
    t_table=CTable(t_table_struct=sqlite3_tables["features_and_return"]),
    t_remove_existence=run_mode in ["O", "OVERWRITE"]
)

for trade_date in calendar.get_iter_list(bgn_date, stp_date, True):
    save_date_dir = os.path.join(research_features_and_return_dir, trade_date[0:4], trade_date)
    for equity_index_code, equity_instru_id in equity_indexes:
        if trade_date <= "20220722" and equity_instru_id == "IM.CFE":
            continue

        features_and_return_file = "{}-{}-features_and_return.csv.gz".format(trade_date, equity_instru_id)
        features_and_return_path = os.path.join(save_date_dir, features_and_return_file)
        features_and_ret_df = pd.read_csv(features_and_return_path, dtype={"trade_date": str, "timestamp": int})

        if run_mode in ["A", "APPEND"]:
            print(features_and_ret_df)
            features_and_return_lib.delete_by_date(t_date=trade_date)
        features_and_return_lib.update_by_date(
            t_date=trade_date,
            t_update_df=features_and_ret_df,
        )

    print("... @ {0}, features and return of {1} converted to sqlite3".format(dt.datetime.now(), trade_date))

features_and_return_lib.close()
