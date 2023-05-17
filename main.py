import subprocess as sp

md_bgn_date, md_stp_date = "20160101", "20230501"

sp.run(["python", "00_features_and_return.py", md_bgn_date, md_stp_date])
sp.run(["python", "01_convert_csv_to_sqlite3.py",
        "--mode", "o",
        "--bgn", md_bgn_date,
        "--stp", md_stp_date])
