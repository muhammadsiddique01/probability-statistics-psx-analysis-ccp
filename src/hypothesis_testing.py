from scipy.stats import ttest_1samp

def run_ttest(df):

    print("\n==========================")
    print("HYPOTHESIS TESTING")
    print("==========================")

    results = {}

    for company in df["Symbol"].unique():

        company_df = df[df["Symbol"] == company]

        returns = company_df["Close"].pct_change().dropna()

        t_stat, p_value = ttest_1samp(returns, 0)

        results[company] = {
            "t_statistic": t_stat,
            "p_value": p_value
        }

        print(f"\n{company}")
        print("T Statistic:", round(t_stat,4))
        print("P Value:", round(p_value,4))

    return results