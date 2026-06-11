from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

def run_regression(df):

    print("\n==========================")
    print("LINEAR REGRESSION")
    print("==========================")

    for company in df["Symbol"].unique():

        company_df = df[df["Symbol"] == company]

        X = company_df[["Open","High","Low","Volume"]]

        y = company_df["Close"]

        X_train,X_test,y_train,y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        model = LinearRegression()

        model.fit(X_train,y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test,predictions)

        r2 = r2_score(y_test,predictions)

        print(f"\n{company}")
        print("MAE:",round(mae,4))
        print("R2 Score:",round(r2,4))