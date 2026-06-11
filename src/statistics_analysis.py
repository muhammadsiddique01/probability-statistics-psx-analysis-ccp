import pandas as pd

def descriptive_statistics(df):

    print("\n==========================")
    print("DESCRIPTIVE STATISTICS")
    print("==========================")

    stats = df.groupby("Symbol")["Close"].agg(
        Mean="mean",
        Median="median",
        Variance="var",
        Std_Dev="std",
        Minimum="min",
        Maximum="max"
    )

    print(stats)

    stats.to_csv("result/descriptive_statistics.csv")

    return stats