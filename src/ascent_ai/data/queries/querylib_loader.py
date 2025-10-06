import os
import glob
from datetime import datetime


def get_latest_querylib_file(main_path):
    querylib_files = glob.glob(os.path.join(main_path, "querylib_*.db"))
    querylib_files.sort(
        key=lambda filename: datetime.strptime(
            filename.split("_")[-1].split(".")[0], "%Y%m%d"
        ),
        reverse=True,
    )
    return querylib_files[0] if querylib_files else None
