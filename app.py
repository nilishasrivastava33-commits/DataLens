import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="DataLens", page_icon="", layout="wide")

st.title("DataLens")
st.subheader("Automated Data Analysis & Insight Platform")
st.write(
    "Upload a CSV file to explore data quality, clean the data, "
    "view statistics and charts, discover simple insights, and try a machine-learning demo."
)

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read this CSV file: {exc}")
    st.stop()

if df.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()

st.success(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns.")

def clean_dataset(data):
    cleaned = data.copy()
    original_rows = len(cleaned)
    cleaned = cleaned.drop_duplicates()

    numeric_cols = cleaned.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        median = cleaned[col].median()
        if pd.notna(median):
            cleaned[col] = cleaned[col].fillna(median)

    categorical_cols = cleaned.select_dtypes(
        include=["object", "category", "string"]
    ).columns
    for col in categorical_cols:
        modes = cleaned[col].mode(dropna=True)
        if not modes.empty:
            cleaned[col] = cleaned[col].fillna(modes.iloc[0])

    return cleaned, original_rows - len(cleaned)

def generate_insights(data):
    insights = [
        f"The dataset contains {data.shape[0]:,} rows and {data.shape[1]} columns.",
        f"There are {int(data.duplicated().sum())} duplicate rows.",
        f"There are {int(data.isna().sum().sum())} missing cells."
    ]

    numeric = data.select_dtypes(include=np.number)

    for col in numeric.columns[:6]:
        if numeric[col].notna().any():
            insights.append(
                f"{col}: mean {numeric[col].mean():.2f}; "
                f"median {numeric[col].median():.2f}."
            )

    if numeric.shape[1] >= 2:
        corr = numeric.corr()
        abs_corr = corr.abs().copy()

        # Ignore each column's correlation with itself
        for col in abs_corr.columns:
            abs_corr.loc[col, col] = 0

        stacked = abs_corr.stack()

        if not stacked.empty and stacked.notna().any():
            pair = stacked.idxmax()
            value = corr.loc[pair[0], pair[1]]

            insights.append(
                f"Strongest absolute numerical correlation found: "
                f"{pair[0]} and {pair[1]} "
                f"(correlation {value:.2f}). "
                "Correlation does not prove causation."
            )

    return insights

overview_tab, quality_tab, cleaning_tab, stats_tab, charts_tab, insights_tab, ml_tab = st.tabs([
    "Dataset Overview", "Data Quality", "Clean Data", "Statistics",
    "Visualizations", "Automatic Insights", "Machine Learning"
])

with overview_tab:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    c4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    st.markdown("#### Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("#### Column names and data types")
    type_table = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })
    st.dataframe(type_table, use_container_width=True)

with quality_tab:
    report = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isna().sum().values,
        "Missing Percentage": (df.isna().mean().values * 100).round(2),
        "Unique Values": df.nunique(dropna=True).values
    })
    st.dataframe(report, use_container_width=True)

cleaned_df, duplicates_removed = clean_dataset(df)

with cleaning_tab:
    st.write("This basic cleaning routine removes duplicate rows, fills numerical missing values with the median, and fills categorical missing values with the most frequent value.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Original rows", f"{len(df):,}")
    c2.metric("Duplicate rows removed", f"{duplicates_removed:,}")
    c3.metric("Missing cells after cleaning", f"{int(cleaned_df.isna().sum().sum()):,}")
    st.dataframe(cleaned_df.head(10), use_container_width=True)
    st.download_button(
        "Download cleaned CSV",
        data=cleaned_df.to_csv(index=False).encode("utf-8"),
        file_name="datalens_cleaned.csv",
        mime="text/csv"
    )

with stats_tab:
    numeric_df = cleaned_df.select_dtypes(include=np.number)
    if numeric_df.empty:
        st.warning("No numerical columns were found.")
    else:
        st.dataframe(numeric_df.describe().T.round(2), use_container_width=True)

with charts_tab:
    numeric_cols = cleaned_df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.warning("No numerical columns are available for charts.")
    else:
        selected_col = st.selectbox("Choose a numerical column for its histogram", numeric_cols)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(cleaned_df[selected_col].dropna(), bins=20, edgecolor="black")
        ax.set_title(f"Distribution of {selected_col}")
        ax.set_xlabel(selected_col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        plt.close(fig)

        if len(numeric_cols) >= 2:
            corr = cleaned_df[numeric_cols].corr()
            fig2, ax2 = plt.subplots(figsize=(max(6, len(numeric_cols)), 5))
            image = ax2.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
            ax2.set_xticks(range(len(corr.columns)))
            ax2.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax2.set_yticks(range(len(corr.columns)))
            ax2.set_yticklabels(corr.columns)
            ax2.set_title("Correlation Heatmap")
            fig2.colorbar(image, ax=ax2)
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        for col in numeric_cols[:3]:
            fig3, ax3 = plt.subplots(figsize=(7, 2.5))
            ax3.boxplot(cleaned_df[col].dropna())
            ax3.set_title(f"Box Plot: {col}")
            st.pyplot(fig3)
            plt.close(fig3)

with insights_tab:
    st.write("These are rule-based summaries generated from the uploaded data.")
    for insight in generate_insights(cleaned_df):
        st.write("• " + insight)

with ml_tab:
    st.write(
        "Train a Random Forest regression model to estimate a selected numerical column. "
        "This is a demonstration, not a guarantee of future prediction accuracy."
    )
    numeric_targets = cleaned_df.select_dtypes(include=np.number).columns.tolist()
    # Exclude columns whose names suggest they are identifiers.
    target_options = [
        col for col in numeric_targets
        if "id" not in col.lower() and cleaned_df[col].nunique(dropna=True) > 1
    ]

    if len(target_options) < 1:
        st.warning("No suitable numerical target columns were found.")
    else:
        target = st.selectbox("Select the numerical column to predict", target_options)
        if st.button("Train Prediction Model"):
            model_data = cleaned_df.dropna(subset=[target]).copy()
            y = model_data[target]
            X = model_data.drop(columns=[target])

            # Remove ID-like features to reduce misleading shortcuts.
            id_features = [col for col in X.columns if "id" in col.lower()]
            X = X.drop(columns=id_features, errors="ignore")

            # Avoid target leakage for retail datasets: Total Amount is commonly
            # derived from Quantity and Price per Unit.
            if target.strip().lower() in {"total amount", "total_sales", "sales_amount"}:
                leak_cols = [
                    col for col in X.columns
                    if col.strip().lower() in {"quantity", "price per unit", "price", "unit price"}
                ]
                X = X.drop(columns=leak_cols, errors="ignore")
                if leak_cols:
                    st.info(
                        "Excluded Quantity/Price columns from this demonstration because "
                        "they can directly determine Total Amount and produce misleadingly perfect scores."
                    )

            X = X.dropna(axis=1, how="all")
            if X.empty:
                st.error("No usable features remain after excluding the target and ID columns.")
            elif len(model_data) < 10:
                st.error("At least 10 rows with a valid target are required.")
            else:
                num_features = X.select_dtypes(include=np.number).columns.tolist()
                cat_features = X.select_dtypes(exclude=np.number).columns.tolist()
                transformers = []

                if num_features:
                    transformers.append((
                        "numeric",
                        Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                        num_features
                    ))
                if cat_features:
                    transformers.append((
                        "categorical",
                        Pipeline([
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore"))
                        ]),
                        cat_features
                    ))

                if not transformers:
                    st.error("No usable features remain for training.")
                else:
                    preprocess = ColumnTransformer(transformers=transformers)
                    model = Pipeline([
                        ("preprocess", preprocess),
                        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
                    ])
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    mae = mean_absolute_error(y_test, preds)
                    r2 = r2_score(y_test, preds)
                    a, b = st.columns(2)
                    a.metric("Mean Absolute Error", f"{mae:.2f}")
                    b.metric("R² score", f"{r2:.3f}")
                    st.write(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")
                    st.caption(
                        "R² can be negative. Evaluate this score in context; a high score does not by itself prove the model will predict future data well."
                    )

st.divider()
st.caption("DataLens • Built with Python, Pandas, NumPy, Matplotlib, Scikit-learn and Streamlit.")
