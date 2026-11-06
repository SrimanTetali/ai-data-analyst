
import pandas as pd
import numpy as np


# Supported operations
SUPPORTED_OPERATIONS = {
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "count",
    "group_sum",
    "group_mean",
    "group_count",
    "top_n",
    "bottom_n",
    "correlation",
    "summary",
    "trend",
    "anomalies",
}


def validate_column(df, column):
    """Validate that a column exists in the dataset."""

    if not isinstance(column, str) or column not in df.columns:
        raise ValueError(f"Invalid column: {column}")

    return column


def validate_numeric_column(df, column):
    """Validate that a column contains numeric data."""

    validate_column(df, column)

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            f"Column '{column}' must contain numeric data."
        )

    return column


def get_numeric_columns(df):
    """Return all numeric columns."""

    return df.select_dtypes(include="number").columns.tolist()


# --------------------------------------------------
# Basic Statistical Operations
# --------------------------------------------------

def calculate_sum(df, column):
    validate_numeric_column(df, column)

    return df[column].sum()


def calculate_mean(df, column):
    validate_numeric_column(df, column)

    return df[column].mean()


def calculate_median(df, column):
    validate_numeric_column(df, column)

    return df[column].median()


def calculate_min(df, column):
    validate_numeric_column(df, column)

    return df[column].min()


def calculate_max(df, column):
    validate_numeric_column(df, column)

    return df[column].max()


def calculate_count(df, column):
    validate_column(df, column)

    return df[column].count()


# --------------------------------------------------
# Group-by Operations
# --------------------------------------------------

def group_sum(df, group_by, value_column):
    validate_column(df, group_by)
    validate_numeric_column(df, value_column)

    return (
        df.groupby(group_by, dropna=False)[value_column]
        .sum()
        .reset_index()
        .sort_values(value_column, ascending=False)
    )


def group_mean(df, group_by, value_column):
    validate_column(df, group_by)
    validate_numeric_column(df, value_column)

    return (
        df.groupby(group_by, dropna=False)[value_column]
        .mean()
        .reset_index()
        .sort_values(value_column, ascending=False)
    )


def group_count(df, group_by, value_column=None):
    validate_column(df, group_by)

    if value_column is None:
        result = df.groupby(group_by, dropna=False).size()
    else:
        validate_column(df, value_column)

        result = (
            df.groupby(group_by, dropna=False)[value_column]
            .count()
        )

    return (
        result.rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )


# --------------------------------------------------
# Top / Bottom N
# --------------------------------------------------

def top_n(df, value_column, n=5, group_by=None):
    validate_numeric_column(df, value_column)

    if not isinstance(n, int) or not 1 <= n <= 100:
        raise ValueError("n must be an integer between 1 and 100.")

    columns = [value_column]

    if group_by is not None:
        validate_column(df, group_by)

        if group_by != value_column:
            columns.insert(0, group_by)

    return (
        df[columns]
        .sort_values(value_column, ascending=False)
        .head(n)
    )


def bottom_n(df, value_column, n=5, group_by=None):
    validate_numeric_column(df, value_column)

    if not isinstance(n, int) or not 1 <= n <= 100:
        raise ValueError("n must be an integer between 1 and 100.")

    columns = [value_column]

    if group_by is not None:
        validate_column(df, group_by)

        if group_by != value_column:
            columns.insert(0, group_by)

    return (
        df[columns]
        .sort_values(value_column, ascending=True)
        .head(n)
    )


# --------------------------------------------------
# Correlation Analysis
# --------------------------------------------------

def calculate_correlation(df, columns=None):
    numeric_columns = get_numeric_columns(df)

    if columns is None:
        columns = numeric_columns

    if len(columns) < 2:
        raise ValueError(
            "Correlation requires at least two numeric columns."
        )

    for column in columns:
        validate_numeric_column(df, column)

    return df[columns].corr()


# --------------------------------------------------
# Summary Statistics
# --------------------------------------------------

def summary_statistics(df):
    if df.empty:
        raise ValueError("The dataset is empty.")

    return df.describe(include="all").transpose()


# --------------------------------------------------
# Trend Analysis
# --------------------------------------------------

def trend_analysis(df, date_column, value_column, period="monthly"):
    validate_column(df, date_column)
    validate_numeric_column(df, value_column)

    period_map = {
        "daily": "D",
        "weekly": "W",
        "monthly": "ME",
        "quarterly": "QE",
        "yearly": "YE",
    }

    if period not in period_map:
        raise ValueError(
            "Period must be daily, weekly, monthly, quarterly, or yearly."
        )

    data = df[[date_column, value_column]].copy()

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce"
    )

    data = data.dropna(subset=[date_column])

    if data.empty:
        raise ValueError(
            f"No valid dates found in '{date_column}'."
        )

    result = (
        data.set_index(date_column)
        .resample(period_map[period])[value_column]
        .sum()
        .reset_index()
    )

    return result


# --------------------------------------------------
# Anomaly Detection
# --------------------------------------------------

def detect_anomalies(df, value_column, threshold=3.0):
    validate_numeric_column(df, value_column)

    if not isinstance(threshold, (int, float)) or threshold <= 0:
        raise ValueError("Threshold must be a positive number.")

    data = df.copy()

    values = data[value_column]

    mean = values.mean()
    std = values.std()

    if pd.isna(std) or std == 0:
        return data.iloc[0:0].copy()

    z_scores = (values - mean) / std

    return data.loc[z_scores.abs() > threshold].copy()


# --------------------------------------------------
# Main Analysis Dispatcher
# --------------------------------------------------

def execute_analysis(df, plan):
    """
    Execute a validated analysis plan.

    The LLM should only provide a structured plan.
    It must never provide Python code.
    """

    if not isinstance(plan, dict):
        raise ValueError("Analysis plan must be a dictionary.")

    operation = plan.get("operation")

    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(
            f"Unsupported operation: {operation}"
        )

    if df.empty:
        raise ValueError("The dataset is empty.")

    value_column = plan.get("value_column")
    group_by = plan.get("group_by")

    if operation == "sum":
        return calculate_sum(df, value_column)

    elif operation == "mean":
        return calculate_mean(df, value_column)

    elif operation == "median":
        return calculate_median(df, value_column)

    elif operation == "min":
        return calculate_min(df, value_column)

    elif operation == "max":
        return calculate_max(df, value_column)

    elif operation == "count":
        return calculate_count(df, value_column)

    elif operation == "group_sum":
        return group_sum(df, group_by, value_column)

    elif operation == "group_mean":
        return group_mean(df, group_by, value_column)

    elif operation == "group_count":
        return group_count(df, group_by, value_column)

    elif operation == "top_n":
        return top_n(
            df,
            value_column,
            n=plan.get("n", 5),
            group_by=group_by
        )

    elif operation == "bottom_n":
        return bottom_n(
            df,
            value_column,
            n=plan.get("n", 5),
            group_by=group_by
        )

    elif operation == "correlation":
        return calculate_correlation(
            df,
            columns=plan.get("columns")
        )

    elif operation == "summary":
        return summary_statistics(df)

    elif operation == "trend":
        return trend_analysis(
            df,
            date_column=plan.get("date_column"),
            value_column=value_column,
            period=plan.get("period", "monthly")
        )

    elif operation == "anomalies":
        return detect_anomalies(
            df,
            value_column,
            threshold=plan.get("threshold", 3.0)
        )
