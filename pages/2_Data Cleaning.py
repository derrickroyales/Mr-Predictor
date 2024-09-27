# 2_data_cleaning.py

import streamlit as st
import pandas as pd

# Inject custom CSS for background color
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color:#d3cbb7;  /* Light grey background */
}

[data-testid="stSidebar"] {
    background-color: #fdf6e3;  /* Sidebar background */
}

[data-testid="stHeader"] {
    background-color: transparent;
}

</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.title("Data Cleaning and Feature Selection")

if 'df' not in st.session_state or st.session_state.df is None:
    st.error("Please upload data or connect to a database first.")
else:
    df = st.session_state.df.copy()

    # Data cleaning and conversion functions
    def detect_numeric_columns(df):
        """Ensure that numeric columns are properly detected."""
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_columns:  # If no numeric columns found, force conversion
            df = df.apply(pd.to_numeric, errors='coerce')
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        return numeric_columns

    def convert_columns_to_numeric(df, selected_columns):
        """Convert selected columns to numeric datatype, coercing errors to NaN."""
        for col in selected_columns:
          try:
                # Attempt to convert column to numeric, coercing errors into NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Check if all values in the column were successfully converted
                if df[col].isnull().all():
                    # If entire column becomes NaN, it means conversion failed
                    st.error(f"Column '{col}' could not be converted to numeric. No valid numbers found.")
                elif df[col].isnull().sum() > 0:
                    # Partial failure; warn that some values couldn't be converted
                    st.warning(f"Some values in column '{col}' could not be converted to numbers and were replaced with NaN.")
          except Exception as e:
                st.error(f"An error occurred while converting column '{col}' to numeric: {str(e)}")
        return df

    # Step 3: Data display and conversion of string columns to numeric
    st.write("### Uploaded Data")
    st.dataframe(df.head())

    st.write("### Convert Columns to Numeric Data Type")
    potential_numeric_columns = df.columns[df.dtypes == 'object']

    if not potential_numeric_columns.empty:
        convert_columns = st.multiselect("Select columns to convert to numeric:", potential_numeric_columns)
        if convert_columns:
            df = convert_columns_to_numeric(df, convert_columns)
            st.write("### Data after Conversion")
            st.dataframe(df.head())
    st.divider()

    # Step 4: Handling missing values
    st.write("### Data Cleaning: Handling Missing Values")
    if df.isnull().values.any():
        st.write("The following columns have missing values:")
        missing_columns = df.columns[df.isnull().any()]
        st.write(df[missing_columns].isnull().sum())

        fill_methods = {}
        for col in missing_columns:
            fill_method = st.selectbox(
                f"How would you like to handle missing values in '{col}'?",
                ("Mean", "Median", "Mode", "Fill with value from the upper row","Fill with value from the lower row")
            )
            fill_methods[col] = fill_method

        for col, method in fill_methods.items():
          try:
            if method == "Mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif method == "Median":
                df[col].fillna(df[col].median(), inplace=True)
            elif method == "Mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif method == "Fill with value from the upper row":
                df[col].fillna(method='ffill', inplace=True)
            elif method == "Fill with value from the lower row":
                df[col].fillna(method='bfill', inplace=True)
          except TypeError as e:
                # Custom error message for the user
                st.error(f"Error filling missing values in column '{col}': {str(e)}")
    else:
        st.write("No missing values found.")

    # Step 5: Handling duplicates
    st.write("### Data Cleaning: Handling Duplicates")
    if df.duplicated().any():
        st.write(f"There are {df.duplicated().sum()} duplicate rows.")
        remove_duplicates = st.checkbox("Do you want to remove duplicate rows?", value=False)
        if remove_duplicates:
            df.drop_duplicates(inplace=True)
            st.write("_Duplicates have been removed._")
    else:
        st.write("No duplicate rows found.")
    st.divider()

    # Store cleaned data
    st.session_state.df = df

