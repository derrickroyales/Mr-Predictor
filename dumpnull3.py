import streamlit as st
import pandas as pd
from prophet import Prophet
from io import BytesIO
import sqlalchemy
from sqlalchemy import create_engine

# Inject custom CSS for background color
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color:#d3cbb7;  /* Light grey background */
}

[data-testid="stSidebar"] {
    background-color: #121d52;  /* Sidebar background */
}

[data-testid="stHeader"] {
    background-color: transparent;
}

</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Title of the web app
st.markdown("<h1 style='font-size:12vh; text-align: center; padding-bottom: 0; margin-bottom: 0;color: white;'>Monsieur Prédicteur</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='font-size:2.5vh; text-align: center; font-style: italic;color: white;'>your trusted forecaster</h2>", unsafe_allow_html=True)
st.divider()

# Step 1: Choose between file upload and database connection
data_source = st.radio("Choose data source:", ("Upload a file", "Connect to a database"))

# Initialize data variable
df = None

def load_data(file, file_type):
    """Load data from file based on file type."""
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

# Database Connection logic
if data_source == "Upload a file":
    st.write("### Upload your data file")
    uploaded_file = st.file_uploader("**Upload your data file (Excel, CSV, TXT, XLS)**", type=["xlsx", "csv", "txt", "xls"])

    # If file is uploaded, load the data
    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1]
        df = load_data(uploaded_file, file_type)

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
            # Create a connection to the database
            connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
            engine = create_engine(connection_string)

            # Check if the input is a SQL query or table name
            if db_table_or_query.strip().lower().startswith("select"):
                # Execute the query
                df = pd.read_sql_query(db_table_or_query, engine)
            else:
                # Assume the input is a table name
                df = pd.read_sql_table(db_table_or_query, engine)

            st.success("Data loaded successfully from the database")
            st.dataframe(df.head())  # Show the first few rows of the loaded data

        except Exception as e:
            st.error(f"Error connecting to the database: {str(e)}")

# Proceed with the rest of the steps if data is available (either from upload or database)
if df is not None:
    # Step 2: User inputs for prediction days (Forecast Horizon)
    forecast_horizon_unit = st.selectbox("**Select Forecast Horizon Unit**", ["Days", "Weeks", "Months"])
    forecast_horizon_value = st.number_input(f"**Number of {forecast_horizon_unit} for Prediction**", min_value=1, step=1)

    # Convert the horizon to days for Prophet
    if forecast_horizon_unit == "Days":
        prediction_days = forecast_horizon_value
    elif forecast_horizon_unit == "Weeks":
        prediction_days = forecast_horizon_value * 7
    else:
        prediction_days = forecast_horizon_value * 30

    st.divider()

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
            df[col] = pd.to_numeric(df[col], errors='coerce')
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
                ("Mean", "Median", "Mode", "Fill with value from the upper row")
            )
            fill_methods[col] = fill_method

        for col, method in fill_methods.items():
            if method == "Mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif method == "Median":
                df[col].fillna(df[col].median(), inplace=True)
            elif method == "Mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif method == "Fill with value from the upper row":
                df[col].fillna(method='ffill', inplace=True)
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

    # Step 6: Date and Target column selection
    date_column = st.selectbox("**Select the column to use as Date (ds):**", df.columns)
    target_columns = st.multiselect("**Select the column(s) to predict (target column(s))**:", df.columns)
    df.rename(columns={date_column: 'ds'}, inplace=True)

    selected_features = df.columns.difference(['ds'] + target_columns)

    numeric_columns = detect_numeric_columns(df[selected_features])
    categorical_columns = [col for col in selected_features if df[col].dtype == 'object']
    st.divider()

    # Step 7: Numeric feature selection for the model
    st.write("### Select Numeric Columns for Prediction")
    selected_numeric_columns = st.multiselect("Select numeric columns to use in the prediction:", numeric_columns)
    feature_inputs = {}
    if selected_numeric_columns:
        for col in selected_numeric_columns:
            feature_inputs[col] = st.number_input(f"Enter value for {col}:", value=float(df[col].mean()))

    # Step 8: Categorical feature selection and encoding
    include_categorical = st.checkbox("**Include categorical columns in the prediction?**", value=False)
    categorical_inputs = {}
    selected_categorical_columns = []
    if include_categorical:
        selected_categorical_columns = st.multiselect("Select categorical columns to include:", categorical_columns)
        for cat_col in selected_categorical_columns:
            unique_values = df[cat_col].unique()
            categorical_inputs[cat_col] = st.selectbox(f"Select value for {cat_col}:", unique_values)

        df = pd.get_dummies(df, columns=selected_categorical_columns)
        for cat_col, user_choice in categorical_inputs.items():
            for category in df.columns:
                if category.startswith(cat_col + '_'):
                    feature_inputs[category] = 1 if category == f"{cat_col}_{user_choice}" else 0
    st.divider()

        # Step 9: Seasonal adjustments (User-defined)
    st.write("### Seasonal Adjustments")
    yearly_seasonality = st.checkbox("Include yearly seasonality", value=True)
    weekly_seasonality = st.checkbox("Include weekly seasonality", value=True)
    daily_seasonality = st.checkbox("Include daily seasonality", value=False)

        # Allow the user to define custom seasonalities
    add_custom_seasonality = st.checkbox("Add custom seasonality?", value=False)

    custom_seasonality_name = None
    custom_seasonality_period = None
    custom_seasonality_fourier_order = None

    if add_custom_seasonality:
            custom_seasonality_name = st.text_input("Enter the name of your custom seasonality:")
            custom_seasonality_period = st.number_input("Custom seasonality period (in days):", value=30)
            custom_seasonality_fourier_order = st.number_input("Custom seasonality Fourier order:", value=5)
    st.divider()
        
        
        # Step 11: Train the Prophet model with selected features as regressors
    forecast_data = {}

    for target in target_columns:
            df_temp = df.copy()  # Create a temporary copy of the df to modify per target

            # Rename the target column to 'y' for Prophet
            df_temp.rename(columns={target: 'y'}, inplace=True)

            model = Prophet(
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality
            )

            # Add custom seasonality if defined
            if add_custom_seasonality:
                model.add_seasonality(
                    name=custom_seasonality_name,
                    period=custom_seasonality_period,
                    fourier_order=custom_seasonality_fourier_order
                )

            for feature in feature_inputs:
                model.add_regressor(feature)

            # Fit the model
            model.fit(df_temp[['ds', 'y'] + list(feature_inputs.keys())])

            # Step 12: Make future predictions
            future = model.make_future_dataframe(periods=prediction_days)

            # Add user inputs for numeric and categorical columns to future DataFrame
            for feature in feature_inputs:
                future[feature] = feature_inputs[feature]

            # Make predictions
            forecast = model.predict(future)

            # Step 13: Plot the forecast before renaming any columns
            st.write(f"### Forecast Plot for {target}")
            fig = model.plot(forecast)  # Plot using the Prophet 'forecast' DataFrame with 'ds' and 'yhat'
            st.pyplot(fig)

            # Rename columns back to their original names for display and export (after plotting)
            forecast.rename(columns={'ds': date_column, 'yhat': f'Predicted {target}'}, inplace=True)

            # Store predictions in dictionary for later export
            forecast_data[target] = forecast[[date_column, f'Predicted {target}']].tail(prediction_days)

            # Step 14: Display forecasted data without yhat_lower and yhat_upper
            st.write(f"### Forecast for {target} for the next {forecast_horizon_value} {forecast_horizon_unit.lower()}(s)")
            st.dataframe(forecast_data[target])

        # Step 15: Merge all forecasted DataFrames into one
    merged_forecasts = forecast_data[target_columns[0]].copy()  # Start with the first target's DataFrame
    for target in target_columns[1:]:
            merged_forecasts = pd.merge(merged_forecasts, forecast_data[target], on=date_column, how='inner')

    st.write("### Merged Forecasts for All Targets")
    st.dataframe(merged_forecasts)

        # Step 16: Prepare Excel output with forecast and user-friendly columns
    def to_excel_with_dashboard(merged_forecasts):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Write the merged predictions to an Excel sheet
                merged_forecasts.to_excel(writer, index=False, sheet_name='Merged_Predictions')

                # Optionally, add charts or other details
                workbook = writer.book
                worksheet = writer.sheets['Merged_Predictions']

                # Create a chart for predictions for each target
                for target in target_columns:
                    chart = workbook.add_chart({'type': 'line'})
                    chart.add_series({
                        'name': f'Predicted {target}',
                        'categories': ['Merged_Predictions', 1, 0, len(merged_forecasts), 0],  # Date column
                        'values': ['Merged_Predictions', 1, merged_forecasts.columns.get_loc(f'Predicted {target}'), len(merged_forecasts), merged_forecasts.columns.get_loc(f'Predicted {target}')],  # yhat column
                    })
                    chart.set_x_axis({'name': date_column})
                    chart.set_y_axis({'name': f'Predicted {target}'})
                    chart.set_title({'name': f'Predicted {target} Over Time'})
                    worksheet.insert_chart(f'H{2 + 16 * target_columns.index(target)}', chart)

            processed_data = output.getvalue()
            return processed_data

        # Prepare the merged forecast data for Excel export
    excel_data = to_excel_with_dashboard(merged_forecasts)

        # Step 17: Provide a download button for the Excel file
    st.download_button(
            label="Download Merged Predictions with Charts",
            data=excel_data,
            file_name='merged_predictions_with_dashboard.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )