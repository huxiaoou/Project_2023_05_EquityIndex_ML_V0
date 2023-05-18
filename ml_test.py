import os
import datetime as dt
import sys
from falkreath import CManagerLibReader, CManagerLibWriter, CTable
from whiterun import CCalendarMonthly
from xfuns import read_from_sio_obj


def ml_model_test(model_lbl: str,
                  instrument: str | None, tid: str | None, bgn_date: str, stp_date: str,
                  calendar_path: str,
                  features_and_return_dir: str, models_dir: str, predictions_dir: str,
                  sqlite3_tables: dict,
                  train_windows: list, x_lbls: list, y_lbls: list,
                  ):
    """

    :param model_lbl: ["lm", "nn"]
    :param instrument: like IC.CFE
    :param tid: T01,...,T07
    :param bgn_date: format = [YYYYMMDD]
    :param stp_date: format = [YYYYMMDD], can be skip, and program will use bgn only
    :param calendar_path:
    :param features_and_return_dir:
    :param models_dir:
    :param predictions_dir：
    :param sqlite3_tables:
    :param train_windows:
    :param x_lbls:
    :param y_lbls: "rtm" must be in it
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

    # --- load lib writer
    pred_lib_manager: dict[str, CManagerLibWriter] = {}
    for trn_win in train_windows:
        pred_id = model_grp_id + "-TMW{:02d}".format(trn_win) + "-pred-{}".format(model_lbl)
        predictions_lib = CManagerLibWriter(
            t_db_save_dir=predictions_dir,
            t_db_name=pred_id + ".db",
        )
        predictions_lib_stru = sqlite3_tables[pred_id]
        predictions_lib_tab = CTable(t_table_struct=predictions_lib_stru)
        predictions_lib.initialize_table(predictions_lib_tab)
        pred_lib_manager[pred_id] = predictions_lib

    # --- dates
    iter_months = calendar.map_iter_dates_to_iter_months(bgn_date, stp_date)

    # --- main core
    for train_end_month in iter_months:
        model_month_dir = os.path.join(models_dir, train_end_month[0:4], train_end_month)
        test_month = calendar.get_next_month(train_end_month, 1)
        test_bgn_date, test_end_date = calendar.get_first_date_of_month(test_month), calendar.get_last_date_of_month(test_month)
        for trn_win in train_windows:
            conds = init_conds + [
                ("trade_date", ">=", test_bgn_date),
                ("trade_date", "<=", test_end_date),
            ]
            src_df = features_and_return_lib.read_by_conditions(
                t_conditions=conds,
                t_value_columns=["trade_date", "instrument", "contract", "tid", "timestamp"] + x_lbls + y_lbls
            )
            x_df, y_df = src_df[x_lbls], src_df[y_lbls]

            # --- normalize
            scaler_path = os.path.join(
                model_month_dir,
                "{}_{}_TMW{:02d}.scl".format(model_grp_id, train_end_month, trn_win)
            )
            try:
                scaler = read_from_sio_obj(scaler_path)
            except FileNotFoundError:
                continue
            x_test = scaler.transform(x_df)

            # --- fit model
            if model_lbl.upper() == "LM":
                model_month_file = "{}_{}_TMW{:02d}.lm".format(model_grp_id, train_end_month, trn_win)
            elif model_lbl.upper() == "NN":
                model_month_file = "{}_{}_TMW{:02d}.nn".format(model_grp_id, train_end_month, trn_win)
            else:
                print("... model_lbl = {} is illegal".format(model_lbl))
                print("... please check again")
                print("... this program will terminate at once")
                sys.exit()

            train_model_path = os.path.join(model_month_dir, model_month_file)
            train_model = read_from_sio_obj(train_model_path)
            src_df["pred"] = train_model.predict(X=x_test)[:, 0]

            pred_id = model_grp_id + "-TMW{:02d}".format(trn_win) + "-pred-{}".format(model_lbl)
            pred_lib_manager[pred_id].update(
                t_update_df=src_df[["trade_date", "instrument", "contract", "tid", "timestamp", "rtm", "pred"]],
                t_using_index=False
            )

    features_and_return_lib.close()
    for _ in pred_lib_manager.values():
        _.close()
    return 0
