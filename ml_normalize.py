import os
import datetime as dt
from sklearn.preprocessing import StandardScaler
from falkreath import CManagerLibReader, CTable
from whiterun import CCalendarMonthly
from winterhold import check_and_mkdir
from xfuns import save_to_sio_obj


def ml_normalize(instrument: str | None, tid: str | None, bgn_date: str, stp_date: str,
                 calendar_path: str,
                 features_and_return_dir: str, models_dir: str,
                 features_and_return_db_name: str,
                 features_and_return_db_stru: dict,
                 train_windows: list, x_lbls: list, y_lbls: list,
                 minimum_data_size: int = 100
                 ):
    """

    :param instrument: like IC.CFE
    :param tid: T01,...,T07
    :param bgn_date: format = [YYYYMMDD]
    :param stp_date: format = [YYYYMMDD], can be skip, and program will use bgn only
    :param calendar_path:
    :param features_and_return_dir:
    :param models_dir:
    :param features_and_return_db_name:
    :param features_and_return_db_stru:
    :param train_windows:
    :param x_lbls:
    :param y_lbls:
    :param minimum_data_size:
    :return:
    """

    init_conds = [(k, "=", v) for k, v in zip(("instrument", "tid"), (instrument, tid)) if v is not None]
    model_grp_id = "-".join(["M"] + list(filter(lambda z: z, [instrument, tid])))

    if stp_date is None:
        stp_date = (dt.datetime.strptime(bgn_date, "%Y%m%d") + dt.timedelta(days=1)).strftime("%Y%m%d")

    # --- load calendar
    calendar = CCalendarMonthly(calendar_path)

    # --- load lib writer
    features_and_return_lib = CManagerLibReader(
        t_db_save_dir=features_and_return_dir,
        t_db_name=features_and_return_db_name
    )
    features_and_return_tab = CTable(t_table_struct=features_and_return_db_stru)
    features_and_return_lib.set_default(features_and_return_tab.m_table_name)

    # --- dates
    iter_months = calendar.map_iter_dates_to_iter_months(bgn_date, stp_date)

    # --- main core
    scaler = StandardScaler()
    for train_end_month in iter_months:
        check_and_mkdir(os.path.join(models_dir, train_end_month[0:4]))
        check_and_mkdir(model_month_dir := os.path.join(models_dir, train_end_month[0:4], train_end_month))

        for trn_win in train_windows:
            train_bgn_date, train_end_date = calendar.get_bgn_and_end_dates_for_trailing_window(train_end_month, trn_win)
            conds = init_conds + [
                ("trade_date", ">=", train_bgn_date),
                ("trade_date", "<=", train_end_date),
            ]
            src_df = features_and_return_lib.read_by_conditions(
                t_conditions=conds,
                t_value_columns=["trade_date", "contract", "tid"] + x_lbls + y_lbls
            )

            if len(src_df) < minimum_data_size:
                print("... {} | LR | {:>12s} | {} | M{:02} | size of train data = {:>4d}, not enough data to train |".format(
                    dt.datetime.now(), model_grp_id, train_end_month, trn_win, len(src_df)))
                continue

            x_df, y_df = src_df[x_lbls], src_df[y_lbls]

            # --- normalize
            scaler.fit(x_df)
            scaler_path = os.path.join(
                model_month_dir,
                "{}_{}_TMW{:02d}.scl".format(model_grp_id, train_end_month, trn_win)
            )
            save_to_sio_obj(scaler, scaler_path)

            print("... {} | LR | {:>12s} | {} | M{:02} | Normalized |".format(
                dt.datetime.now(), model_grp_id, train_end_month, trn_win))

    features_and_return_lib.close()
    return 0
