import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="PSX Statistical Analysis",
    layout="wide"
)

st.title("Comparative Probability and Statistical Analysis of Selected PSX Companies")

# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():
    df = pd.read_csv(
        "dataset/psx_final_dataset.csv",
        parse_dates=["Date"],
        dayfirst=True
    )
    return df

df = load_data()

# =========================
# SIDEBAR
# =========================

companies = sorted(df["Symbol"].unique())

selected_company = st.sidebar.selectbox(
    "Select Section",
    ["ALL"] + list(companies) + ["Documentation"]
)

# ==================================================
# ALL COMPANIES VIEW
# ==================================================

if selected_company == "ALL":

    st.header("Dataset Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))
    col2.metric("Companies", df["Symbol"].nunique())
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Company Comparison")

    comparison_data = []

    for company in companies:

        temp = df[df["Symbol"] == company].copy()
        temp["Daily_Return"] = temp["Close"].pct_change()

        probability_up = (
            (temp["Daily_Return"] > 0).sum()
            /
            temp["Daily_Return"].dropna().shape[0]
        )

        comparison_data.append({
            "Company": company,
            "Mean Close": round(temp["Close"].mean(), 2),
            "Std Dev": round(temp["Close"].std(), 2),
            "Probability Increase (%)": round(probability_up * 100, 2)
        })

    comparison_df = pd.DataFrame(comparison_data)

    st.dataframe(comparison_df)

    # =====================
    # MEAN PRICE COMPARISON
    # =====================

    st.subheader("Mean Closing Price Comparison")

    fig, ax = plt.subplots(figsize=(8, 5))

    comparison_df.plot(
        x="Company",
        y="Mean Close",
        kind="bar",
        ax=ax
    )

    st.pyplot(fig)

    # =====================
    # BOXPLOT
    # =====================

    st.subheader("Company Comparison Boxplot")

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    boxplot_data = [
        df[df["Symbol"] == company]["Close"]
        for company in companies
    ]

    ax2.boxplot(
        boxplot_data,
        tick_labels=companies
    )

    ax2.set_ylabel("Closing Price")

    st.pyplot(fig2)

    # =====================
    # PROBABILITY COMPARISON
    # =====================

    st.subheader("Probability Comparison")

    fig_prob, ax_prob = plt.subplots(figsize=(8, 5))

    ax_prob.bar(
        comparison_df["Company"],
        comparison_df["Probability Increase (%)"]
    )

    ax_prob.set_ylabel("Probability (%)")
    ax_prob.set_title("Probability of Price Increase")

    st.pyplot(fig_prob)

# ==================================================
# DOCUMENTATION VIEW
# ==================================================

elif selected_company == "Documentation":

    st.header("Project Documentation")


    st.image("graphs/Flowchart.png", use_container_width=True)


    st.image("graphs/UML.png", use_container_width=True)

    st.success(
        "Project Documentation Loaded Successfully"
    )

# ==================================================
# SINGLE COMPANY VIEW
# ==================================================

else:

    company_df = df[df["Symbol"] == selected_company].copy()

    # Overview
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Records", len(company_df))
    col2.metric("Missing Values", int(company_df.isnull().sum().sum()))
    col3.metric("Duplicate Rows", int(company_df.duplicated().sum()))

    st.subheader("Dataset Preview")
    st.dataframe(company_df.head(10))

    # Descriptive Statistics
    st.header("Descriptive Statistics")

    stats_df = pd.DataFrame({
        "Statistic": ["Mean", "Median", "Variance", "Std Dev", "Minimum", "Maximum"],
        "Value": [
            company_df["Close"].mean(),
            company_df["Close"].median(),
            company_df["Close"].var(),
            company_df["Close"].std(),
            company_df["Close"].min(),
            company_df["Close"].max()
        ]
    })

    stats_df["Value"] = stats_df["Value"].round(2)
    st.table(stats_df)

    # Probability Analysis
    company_df["Daily_Return"] = company_df["Close"].pct_change()

    st.header("Probability Analysis")

    increase_days  = (company_df["Daily_Return"] > 0).sum()
    total_days     = company_df["Daily_Return"].dropna().shape[0]
    probability_up = increase_days / total_days

    st.metric("Probability of Price Increase", f"{probability_up:.2%}")

    avg_volume  = company_df["Volume"].mean()
    high_volume = company_df[company_df["Volume"] > avg_volume]

    conditional_probability = (
        (high_volume["Daily_Return"] > 0).sum() / len(high_volume)
    )

    st.metric(
        "Probability of Increase When Volume Is Above Average",
        f"{conditional_probability:.2%}"
    )

    # Hypothesis Testing
    st.header("Hypothesis Testing")

    returns = company_df["Daily_Return"].dropna()
    t_stat, p_value = ttest_1samp(returns, 0)

    st.write("T Statistic:", round(t_stat, 4))
    st.write("P Value:",     round(p_value, 4))

    if p_value < 0.05:
        st.success("Reject H0")
    else:
        st.warning("Fail to Reject H0")

    # Price Trend
    st.header("Price Trend")

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(company_df["Date"], company_df["Close"])

    # FIX 2: plt.xticks(rotation=45) was at zero indentation — outside the
    # else block entirely. Python raised IndentationError because the next
    # line (ax3.set_title) was indented but this line was not.
    # Moved inside the block with correct indentation.
    plt.xticks(rotation=45)
    ax3.set_title(f"{selected_company} Closing Price Trend")
    ax3.set_ylabel("Close Price")
    st.pyplot(fig3)

    # Return Distribution
    st.header("Return Distribution")

    fig4, ax4 = plt.subplots(figsize=(10, 5))
    returns.hist(ax=ax4)
    st.pyplot(fig4)

    # Volume vs Close
    st.header("Volume vs Close Price")

    fig5, ax5 = plt.subplots(figsize=(8, 5))
    ax5.scatter(company_df["Volume"], company_df["Close"], alpha=0.5)
    ax5.set_xlabel("Volume")
    ax5.set_ylabel("Close Price")
    st.pyplot(fig5)

    # Linear Regression
    st.header("Linear Regression Prediction")

    X = company_df[["Open", "High", "Low", "Volume"]]
    y = company_df["Close"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    r2  = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    col1, col2 = st.columns(2)
    col1.metric("R² Score", round(r2, 4))
    col2.metric("MAE",      round(mae, 4))

    # Actual vs Predicted
    st.subheader("Actual vs Predicted Prices")

    fig6, ax6 = plt.subplots(figsize=(8, 5))
    ax6.scatter(y_test, predictions, alpha=0.5)
    ax6.set_xlabel("Actual Prices")
    ax6.set_ylabel("Predicted Prices")
    ax6.set_title("Actual vs Predicted")
    st.pyplot(fig6)

    st.success("Dashboard Loaded Successfully")