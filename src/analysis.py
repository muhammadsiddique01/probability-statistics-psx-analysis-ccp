import pandas as pd

def probability_analysis(df):

    print("\n==========================")
    print("PROBABILITY ANALYSIS")
    print("==========================")

    for company in df["Symbol"].unique():

        company_df = df[df["Symbol"] == company].copy()

        company_df["Daily_Return"] = (
            company_df["Close"].pct_change()
        )

        increase_days = (
            company_df["Daily_Return"] > 0
        ).sum()

        total_days = (
            company_df["Daily_Return"]
            .dropna()
            .shape[0]
        )

        probability = increase_days / total_days

        print(
            f"{company}: "
            f"{probability:.2%}"
        )