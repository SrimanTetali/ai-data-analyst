
import streamlit as st
import pandas as pd

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

    except Exception as e:

        st.error(
            f"Error loading the selected file: {e}"
        )

else:

    st.info(
        "Upload one or more CSV or Excel files to get started."
    )
