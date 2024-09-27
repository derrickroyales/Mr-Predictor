import streamlit as st
import pandas as pd
from prophet import Prophet
from io import BytesIO

# Title of the web app
st.title("Sales Prediction Web App with Data Cleaning")

# Step 1: File upload
st.write("### Upload your data file:")
uploaded_file = st.file_uploader("Upload your data file (Excel, CSV, TXT, XLS)", type=["xlsx", "csv", "txt", "xls"])

# Step 2: User inputs for prediction days
prediction_days = st.number_input("Number of Days for Prediction:", min_value=1, step=1)

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

def detect_numeric_columns(df):
    """Ensure that numeric columns are properly detected."""
    # Attempt to convert non-numeric columns to numeric, forcing errors to NaN
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_columns:  # If no numeric columns found, force conversion
        df = df.apply(pd.to_numeric, errors='coerce')
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        
    return numeric_columns

if uploaded_file is not None and prediction_days:
    file_type = uploaded_file.name.split('.')[-1]
    df = load_data(uploaded_file, file_type)

    if df is not None:
        # Display the uploaded data
        st.write("### Uploaded Data")
        st.dataframe(df.head())

        # Step 4: Check for missing values
        st.write("### Data Cleaning: Handling Missing Values")

        if df.isnull().values.any():
            st.write("The following columns have missing values:")
            missing_columns = df.columns[df.isnull().any()]
            st.write(df[missing_columns].isnull().sum())

            fill_methods = {}  # Dictionary to store user choices for each column

            for col in missing_columns:
                fill_method = st.selectbox(
                    f"How would you like to handle missing values in '{col}'?",
                    ("Mean", "Median", "Mode", "Fill with value from the upper row")
                )
                fill_methods[col] = fill_method

            # Apply the selected method for handling missing values
            for col, method in fill_methods.items():
                if method == "Mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif method == "Median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif method == "Mode":
                    df[col].fillna(df[col].mode()[0], inplace=True)  # Fill with the most frequent value
                elif method == "Fill with value from the upper row":
                    df[col].fillna(method='ffill', inplace=True)
        else:
            st.write("No missing values found.")

        # Step 5: Check for duplicates
        st.write("### Data Cleaning: Handling Duplicates")

        if df.duplicated().any():
            st.write(f"There are {df.duplicated().sum()} duplicate rows.")
            remove_duplicates = st.checkbox("Do you want to remove duplicate rows?", value=False)
            if remove_duplicates:
                df.drop_duplicates(inplace=True)
                st.write("Duplicates have been removed.")
        else:
            st.write("No duplicate rows found.")

        # Step 6: Ask user which columns to use as 'ds' and 'y'
        date_column = st.selectbox("Select the column to use as Date (ds):", df.columns)
        target_columns = st.multiselect("Select the column(s) to predict (target column(s)):", df.columns)

        # Rename selected date column for Prophet
        df.rename(columns={date_column: 'ds'}, inplace=True)

        # Step 7: Detect numeric and non-numeric columns for input
        selected_features = df.columns.difference(['ds'] + target_columns)

        # Detect numeric columns more effectively
        numeric_columns = detect_numeric_columns(df[selected_features])
        categorical_columns = [col for col in selected_features if df[col].dtype == 'object']

        feature_inputs = {}  # To store user inputs for numeric columns

        # Step 8: Get user inputs for numeric columns
        st.write("### Numeric Feature Inputs")
        if numeric_columns:
            for col in numeric_columns:
                feature_inputs[col] = st.number_input(f"Enter value for {col}:", value=float(df[col].mean()))
        else:
            st.write("No numeric columns detected.")

        # Step 9: Ask if the user wants to include categorical columns
        include_categorical = st.checkbox("Include categorical columns in the prediction?", value=False)

        # Step 10: If the user chooses to include categorical columns, allow them to select which ones to include
        categorical_inputs = {}
        selected_categorical_columns = []
        if include_categorical:
            st.write("### Select Categorical Columns to Include")
            selected_categorical_columns = st.multiselect(
                "Select categorical columns to include:", categorical_columns)

            # Display inputs for the selected categorical columns
            for cat_col in selected_categorical_columns:
                unique_values = df[cat_col].unique()  # Get unique values for the dropdown
                categorical_inputs[cat_col] = st.selectbox(f"Select value for {cat_col}:", unique_values)

            # One-hot encode the DataFrame and apply user selections for only the selected categorical columns
            df = pd.get_dummies(df, columns=selected_categorical_columns)

            # Set up user input for categorical columns
            for cat_col, user_choice in categorical_inputs.items():
                for category in df.columns:
                    if category.startswith(cat_col + '_'):
                        if category == f"{cat_col}_{user_choice}":
                            feature_inputs[category] = 1
                        else:
                            feature_inputs[category] = 0

        # Step 11: Train the Prophet model with selected features as regressors
        forecast_data = {}

        for target in target_columns:
            df_temp = df.copy()  # Create a temporary copy of the df to modify per target

            # Rename the target column to 'y' for Prophet
            df_temp.rename(columns={target: 'y'}, inplace=True)

            model = Prophet()

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
            st.write(f"### Forecast for {target} for the next {prediction_days} days")
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