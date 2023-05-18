equity_indexes = (
    ("000016.SH", "IH.CFE"),
    ("000300.SH", "IF.CFE"),
    ("000905.SH", "IC.CFE"),
    ("000852.SH", "IM.CFE"),
)

sqlite3_tables = {
    "features_and_return": {
        "table_name": "features_and_return",
        "primary_keys": {
            "trade_date": "TEXT",
            "instrument": "TEXT",
            "contract": "TEXT",
            "tid": "TEXT",
            "timestamp": "INT4",
        },
        "value_columns": {
            "alpha00": "REAL",
            "alpha01": "REAL",
            "alpha02": "REAL",
            "alpha03": "REAL",
            "alpha04": "REAL",
            "alpha05": "REAL",
            "alpha06": "REAL",
            "alpha07": "REAL",
            "alpha08": "REAL",
            "alpha09": "REAL",
            "alpha10": "REAL",
            "alpha11": "REAL",
            "alpha12": "REAL",
            "alpha13": "REAL",
            "alpha14": "REAL",
            "alpha15": "REAL",
            "alpha16": "REAL",
            "alpha17": "REAL",
            "alpha18": "REAL",
            "rtm": "REAL",
        }
    }
}

train_windows = (6, 12, 24)
x_lbls = ["alpha{:02d}".format(_) for _ in range(19)]
y_lbls = ["rtm"]
