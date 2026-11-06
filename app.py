
import streamlit as st
import pandas as pd
from src.analysis_engine import execute_analysis

# Page Configuration
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide"
)

# App Title
st.title("📊 AI Data Analyst")
st.markdown(
    "Upload your CSV or Excel files and explore your data."
)

st.divider()

# File Upload
uploaded_files = st.file_uploader(
    "Upload CSV or Excel files",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
    help="You can upload multiple CSV and Excel files."
)

# Data Loading Function
def load_csv(file):
    """Load CSV with common encoding fallbacks."""

    encodings = ["utf-8-sig", "utf-8", "cp1252"]

    for encoding in encodings:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError("Unable to decode this CSV file.")


def load_excel(file, sheet_name):
    """Load a selected Excel worksheet."""

    file.seek(0)

    return pd.read_excel(
        file,
        sheet_name=sheet_name
    )


# File Selection
if uploaded_files:

    st.success(
        f"{len(uploaded_files)} file(s) uploaded successfully!"
    )

    file_index = st.selectbox(
        "Select a file to analyze",
        options=range(len(uploaded_files)),
        format_func=lambda i: uploaded_files[i].name
    )

    selected_file = uploaded_files[file_index]

    # Load Selected Dataset
    try:

        if selected_file.name.lower().endswith(".csv"):

            df = load_csv(selected_file)

        else:

            selected_file.seek(0)

            excel_file = pd.ExcelFile(selected_file)

            sheet_names = excel_file.sheet_names

            selected_sheet = st.selectbox(
                "Select an Excel worksheet",
                options=sheet_names
            )

            df = load_excel(
                selected_file,
                selected_sheet
            )

        # Dataset Overview
        st.divider()
        st.subheader("Dataset Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Rows", df.shape[0])

        with col2:
            st.metric("Total Columns", df.shape[1])

        with col3:
            missing_values = int(
                df.isnull().sum().sum()
            )
            st.metric("Missing Values", missing_values)

        # Dataset Preview
        st.subheader("Dataset Preview")

        preview_rows = st.slider(
            "Number of rows to preview",
            min_value=5,
            max_value=min(100, max(5, len(df))),
            value=min(10, max(5, len(df)))
        )

        st.dataframe(
            df.head(preview_rows),
            use_container_width=True
        )

        # Column Information
        st.subheader("Column Information")

        column_info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })

        st.dataframe(
            column_info,
            use_container_width=True,
            hide_index=True
        )

        # Summary Statistics
        with st.expander("View Summary Statistics"):

            st.dataframe(
                df.describe(include="all").transpose(),
                use_container_width=True
            )

        
        # -----------------------------
        # Data Analysis
        # -----------------------------
        st.divider()
        st.header("Data Analysis Engine")

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

        all_columns = df.columns.tolist()

        operation = st.selectbox(
            "Select an analysis operation",
            options=[
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
            ]
        )

        plan = {
            "operation": operation
        }

        # -----------------------------
        # Operation Parameters
        # -----------------------------

        if operation in ["sum", "mean", "median", "min", "max"]:
            plan["value_column"] = st.selectbox(
                "Select numeric column",
                numeric_columns
            )

        elif operation == "count":
            plan["value_column"] = st.selectbox(
                "Select column to count",
                all_columns
            )

        elif operation in ["group_sum", "group_mean"]:
            plan["group_by"] = st.selectbox(
                "Group by column",
                all_columns
            )

            plan["value_column"] = st.selectbox(
                "Numeric column",
                numeric_columns
            )

        elif operation == "group_count":
            plan["group_by"] = st.selectbox(
                "Group by column",
                all_columns
            )

            plan["value_column"] = st.selectbox(
                "Column to count",
                ["Count rows"] + all_columns
            )

            if plan["value_column"] == "Count rows":
                plan["value_column"] = None

        elif operation in ["top_n", "bottom_n"]:
            plan["value_column"] = st.selectbox(
                "Numeric column",
                numeric_columns
            )

            plan["n"] = st.slider(
                "Number of records",
                min_value=1,
                max_value=20,
                value=5
            )

        elif operation == "correlation":
            plan["columns"] = st.multiselect(
                "Select numeric columns",
                numeric_columns,
                default=numeric_columns
            )

        elif operation == "trend":
            plan["date_column"] = st.selectbox(
                "Date column",
                all_columns
            )

            plan["value_column"] = st.selectbox(
                "Numeric column",
                numeric_columns
            )

            plan["period"] = st.selectbox(
                "Trend period",
                ["daily", "weekly", "monthly", "quarterly", "yearly"],
                index=2
            )

        elif operation == "anomalies":
            plan["value_column"] = st.selectbox(
                "Numeric column",
                numeric_columns
            )

            plan["threshold"] = st.slider(
                "Anomaly threshold (Z-score)",
                min_value=1.0,
                max_value=5.0,
                value=3.0,
                step=0.5
            )

        # -----------------------------
        # Execute Analysis
        # -----------------------------

        if st.button("Run Analysis", type="primary"):

            try:
                result = execute_analysis(df, plan)

                st.subheader("Analysis Result")

                if isinstance(result, pd.DataFrame):
                    st.dataframe(
                        result,
                        use_container_width=True
                    )

                else:
                    st.metric(
                        label=f"Result: {operation}",
                        value=f"{result:,.2f}"
                        if isinstance(result, (int, float))
                        else str(result)
                    )

            except Exception as e:
                st.error(f"Analysis failed: {e}")


    except Exception as e:

        st.error(
            f"Error loading the selected file: {e}"
        )

else:

    st.info(
        "Upload one or more CSV or Excel files to get started."
    )
