import pandas as pd
from datetime import datetime

def save_attendance(attendance_list):
    # Remove duplicates
    attendance_list = list(set(attendance_list))

    df = pd.DataFrame({
        "Name": attendance_list,
        "Date": datetime.now().date(),
        "Time": datetime.now().time()
    })

    df.to_csv("attendance.csv", mode='a', index=False, header=False)

    print("✅ Attendance Saved")