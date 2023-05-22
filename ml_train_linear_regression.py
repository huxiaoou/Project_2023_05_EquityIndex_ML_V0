import os
import datetime as dt
from sklearn.linear_model import LinearRegression
from falkreath import CManagerLibReader, CTable
from whiterun import CCalendarMonthly
from winterhold import check_and_mkdir
from xfuns import save_to_sio_obj
from xfuns import read_from_sio_obj


def ml_linear_regression(instrument: str | None, tid: str | None, trn_win: int,
                         bgn_date: str, stp_date: str,
                         calendar_path: str,
                         features_and_return_dir: str, models_dir: str,
                         sqlite3_tables: dict,
                         x_lbls: list, y_lbls: list,
                         minimum_data_size: int = 100
                         ):
    """

    :param instrument: like IC.CFE
    :param tid: ['T01',...,'T07']
    :param trn_win: [6,12,24]
    :param bgn_date: format = [YYYYMMDD]
    :param stp_date: format = [YYYYMMDD], can be skip, and program will use bgn only
    :param calendar_path:
    :param features_and_return_dir:
    :param models_dir:
    :param sqlite3_tables:
    :param x_lbls:
    :param y_lbls: "rtm" must be in it
    :param minimum_data_size:
    :return:
    """

    init_conds = [(k, "=", v) for k, v in zip(("instrument", "tid"), (instrument, tid)) if v is not None]
    model_grp_id = "-".join(["M"] + list(filter(lambda z: z, [instrument, tid])))

    if stp_date is None:
        stp_date = (dt.datetime.strptime(bgn_date, "%Y%m%d") + dt.timedelta(days=1)).strftime("%Y%m%d")

    # --- load calendar
    calendar = CCalendarMonthly(calendar_path)

    # --- load lib reader
    features_and_return_lib = CManagerLibReader(
        t_db_save_dir=features_and_return_dir,
        t_db_name="features_and_return.db"
    )
    features_and_return_db_stru = sqlite3_tables["features_and_return"]
    features_and_return_tab = CTable(t_table_struct=features_and_return_db_stru)
    features_and_return_lib.set_default(features_and_return_tab.m_table_name)

    # --- dates
    iter_months = calendar.map_iter_dates_to_iter_months(bgn_date, stp_date)

    # --- main core
    lm = LinearRegression()
    for train_end_month in iter_months:
        check_and_mkdir(os.path.join(models_dir, train_end_month[0:4]))
        check_and_mkdir(model_month_dir := os.path.join(models_dir, train_end_month[0:4], train_end_month))

        train_bgn_date, train_end_date = calendar.get_bgn_and_end_dates_for_trailing_window(train_end_month, trn_win)
        conds = init_conds + [
            ("trade_date", ">=", train_bgn_date),
            ("trade_date", "<=", train_end_date),
        ]
        src_df = features_and_return_lib.read_by_conditions(
            t_conditions=conds,
            t_value_columns=x_lbls + y_lbls
        )

        if len(src_df) < minimum_data_size:
            continue

        x_df, y_df = src_df[x_lbls], src_df[y_lbls]

        # --- normalize
        scaler_path = os.path.join(
            model_month_dir,
            "{}_{}_TMW{:02d}.scl".format(model_grp_id, train_end_month, trn_win)
        )
        scaler = read_from_sio_obj(scaler_path)
        x_train = scaler.transform(x_df)
        # equivalent to: x_train = (x_df - x_df.mean()) / x_df.std(ddof=0)

        # --- fit model
        lm.fit(X=x_train, y=y_df)
        lm_path = os.path.join(
            model_month_dir,
            "{}_{}_TMW{:02d}.lm".format(model_grp_id, train_end_month, trn_win)
        )
        save_to_sio_obj(lm, lm_path)
        r20 = lm.score(X=x_train, y=y_df)
        # # --- validate
        # y = y_df.values[:, 0]
        # y_h = lm.predict(X=x_train)[:, 0]
        # n = len(y)
        # r21 = np.corrcoef(y, y_h)[0, 1] ** 2
        # sst = np.sum((y - y.mean()) ** 2)
        # ssr = np.sum((y_h - y.mean()) ** 2)
        # sse = np.sum((y - y_h) ** 2)
        # e = sst - ssr - sse
        # r22 = 1 - sse / sst

        print("... {} | LR | {:>12s} | {} | M{:02} | R-square = {:.6f} |".format(
            dt.datetime.now(), model_grp_id, train_end_month, trn_win, r20))

    features_and_return_lib.close()
    return 0
