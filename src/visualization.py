import matplotlib.pyplot as plt

def create_graphs(df):

    companies = df["Symbol"].unique()

    # Mean Closing Price Comparison

    mean_prices = df.groupby("Symbol")["Close"].mean()

    plt.figure(figsize=(8,5))
    mean_prices.plot(kind="bar")
    plt.title("Mean Closing Price Comparison")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig("graphs/mean_price_comparison.png")
    plt.close()

    # Boxplot

    plt.figure(figsize=(8,5))

    data = [
        df[df["Symbol"] == company]["Close"]
        for company in companies
    ]

    plt.boxplot(data, labels=companies)

    plt.title("Stock Price Distribution")

    plt.tight_layout()

    plt.savefig("graphs/boxplot_comparison.png")

    plt.close()

    print("\nGraphs Saved Successfully")