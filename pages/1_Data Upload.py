# 1_data_upload.py

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

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

st.title("Upload or Connect to Data")

if 'df' not in st.session_state:
    st.session_state.df = None
if 'forecast_horizon_value'not in st.session_state:
    st.session_state.forecast_horizon_value = 1
if 'data_source' not in st.session_state:
    st.session_state.data_source = "Upload a file"

# Step 1: Choose between file upload and database connection
data_source = st.radio("Choose data source:", ("Upload a file", "Connect to a database"), index=["Upload a file", "Connect to a database"].index(st.session_state.data_source))
st.session_state.data_source = data_source

# Load data function
def load_data(file, file_type):
    if file_type == "xlsx" or file_type == "xls":
        df = pd.read_excel(file)
    elif file_type == "csv":
        df = pd.read_csv(file)
    elif file_type == "txt":
        delimiter = st.text_input("Enter delimiter for the text file:", value=",")
        df = pd.read_csv(file, delimiter=delimiter)
    else:
        st.error("Unsupported file type")
        return None
    return df

if data_source == "Upload a file":
    st.write("### Upload your data file")
    uploaded_file = st.file_uploader("Upload your data file (Excel, CSV, TXT, XLS)", type=["xlsx", "csv", "txt", "xls"])

    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1]
        st.session_state.df = load_data(uploaded_file, file_type)

elif data_source == "Connect to a database":
    st.write("### Database Connection Details")

    # User inputs for database credentials
    db_host = st.text_input("Enter the database host", "localhost")
    db_name = st.text_input("Enter the database name", "")
    db_user = st.text_input("Enter the database username", "")
    db_password = st.text_input("Enter the database password", type="password")
    db_table_or_query = st.text_area("Enter the table name or SQL query", "")

    # Button to connect to the database
    connect_button = st.button("Connect to the database")

    if connect_button and db_table_or_query:
        try:
            connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
            engine = create_engine(connection_string)

            if db_table_or_query.strip().lower().startswith("select"):
                st.session_state.df = pd.read_sql_query(db_table_or_query, engine)
            else:
                st.session_state.df = pd.read_sql_table(db_table_or_query, engine)
            st.success("Data loaded successfully from the database")
            st.dataframe(st.session_state.df.head())
        except Exception as e:
            st.error(f"Error connecting to the database: {str(e)}")
