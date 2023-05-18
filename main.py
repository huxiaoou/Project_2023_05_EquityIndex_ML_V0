import subprocess as sp
import itertools as ittl
from project_setup import calendar_path
from project_setup import research_features_and_return_dir
from project_setup import research_models_dir
from project_config import sqlite3_tables
from project_config import train_windows
from project_config import x_lbls, y_lbls
from project_config import instruments_universe, tids
from ml_linear_regression import ml_linear_regression

md_bgn_date, md_stp_date = "20160101", "20230515"
trn_bgn_date, trn_stp_date = "20180101", "20230515"

switch = {
    "features_and_return": False,
    "toSql": False,
    "lm": True
}
if switch["features_and_return"]:
    sp.run(["python", "00_features_and_return.py", md_bgn_date, md_stp_date])

if switch["toSql"]:
    sp.run(["python", "01_convert_csv_to_sqlite3.py",
            "--mode", "o",
            "--bgn", md_bgn_date,
            "--stp", md_stp_date])

if switch["lm"]:
    for instrument, tid in ittl.product(instruments_universe + [None], tids + [None]):
        ml_linear_regression(
            instrument=instrument, tid=tid,
            bgn_date=trn_bgn_date, stp_date=trn_stp_date,
            calendar_path=calendar_path,
            features_and_return_dir=research_features_and_return_dir,
            models_dir=research_models_dir,
            features_and_return_db_name="features_and_return.db",
            features_and_return_db_stru=sqlite3_tables["features_and_return"],
            train_windows=train_windows, x_lbls=x_lbls, y_lbls=y_lbls
        )
