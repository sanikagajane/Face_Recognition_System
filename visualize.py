import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show_graph():
    df = pd.read_csv("attendance.csv")

    count = df['Name'].value_counts()

    sns.barplot(x=count.index, y=count.values)
    plt.title("Attendance Count")
    plt.xlabel("Students")
    plt.ylabel("Days Present")

    plt.show()