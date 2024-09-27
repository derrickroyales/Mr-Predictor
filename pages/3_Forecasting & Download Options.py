# 3_forecasting.py

import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from io import BytesIO

from utils import detect_numeric_columns
from sklearn.preprocessing import LabelEncoder

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

st.title("Forecasting and Download Options")

if 'df' not in st.session_state or st.session_state.df is None:
    st.error("Please upload data or connect to a database first.")
else:
    df = st.session_state.df.copy()

    # Forecast horizon
    forecast_horizon_unit = st.selectbox("Select Forecast Horizon Unit", ["Days", "Weeks", "Months"])
    forecast_horizon_value = st.number_input(f"Number of {forecast_horizon_unit}", min_value=1, step=1)

    # Convert horizon to days for Prophet
    if forecast_horizon_unit == "Days":
        prediction_days = forecast_horizon_value
    elif forecast_horizon_unit == "Weeks":
        prediction_days = forecast_horizon_value * 7
    else:
        prediction_days = forecast_horizon_value * 30

    # Step 6: Date and Target column selection
    date_column = st.selectbox("**Select the column to use as Date (ds):**", df.columns)
    target_columns = st.multiselect("**Select the column(s) to predict (target column(s))**:", df.columns.difference([date_column]))
    df.rename(columns={date_column: 'ds'}, inplace=True)

    selected_features = df.columns.difference(['ds'] + target_columns)

    numeric_columns = detect_numeric_columns(df[selected_features])
    categorical_columns = [col for col in selected_features if df[col].dtype == 'object']
    st.divider()

    # Step 7: Numeric feature selection for the model
    st.write("### Select Numeric Columns for Prediction")
    selected_numeric_columns = st.multiselect("Select numeric columns whose values you wish to remain constant as the prediction occurs:", numeric_columns)
    feature_inputs = {}
    if selected_numeric_columns:
        for col in selected_numeric_columns:
            feature_inputs[col] = st.number_input(f"Enter constant value for {col}:", value=float(df[col].mean()))

    # Step 8: Categorical feature selection and encoding
    include_categorical = st.checkbox("**Include categorical columns in the prediction?**", value=False)
    categorical_inputs = {}
    selected_categorical_columns = []
    if include_categorical:
        selected_categorical_columns = st.multiselect("Select categorical columns whose values you wish to remain constant as the prediction occurs:", categorical_columns)
        for cat_col in selected_categorical_columns:
            unique_values = df[cat_col].unique()
            categorical_inputs[cat_col] = st.selectbox(f"Select the constant value for {cat_col}:", unique_values)

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

    # Plot selection
    plot_type = st.radio("Choose plot type for forecast visualization:", ('Line Plot', 'Area Plot'))
    
    label_encoders = {}

    for target in target_columns:
        df_temp = df.copy()  # Create a temporary copy of the df to modify per target
        
        # Check if the target column is non-numeric
        if df_temp[target].dtype == 'object':
        # Apply Label Encoding for non-numeric target column
         st.warning(f"Target column '{target}' is non-numeric. Applying Label Encoding for conversion.")
         le = LabelEncoder()
         df_temp[target] = le.fit_transform(df_temp[target])
         label_encoders[target] = le  # Store the encoder to reverse the transformation later

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
        
         # Convert the predicted yhat values back to their original categorical form if necessary
        if target in label_encoders:
        # Reverse the label encoding using inverse_transform
         forecast['yhat'] = label_encoders[target].inverse_transform(forecast['yhat'].astype(int))
         
        # Step 13: Plot the forecast before renaming any columns
        st.write(f"### General Plot for {target} with the forecasted values")
        if plot_type == 'Line Plot':
            fig, ax = plt.subplots(figsize=(10, 6))
    
            # Plot the predicted 'yhat' line (Predicted values)
            ax.plot(forecast['ds'], forecast['yhat'], label=f'Predicted {target}', color='blue', linestyle='-', marker=None)
    
            # # Optionally, you can still plot the historical data as a scatter plot if you wish
            # ax.scatter(forecast['ds'], forecast['yhat'], label=f'Actual {target}', color='black', marker='.', alpha=0.5)
    
            # # Adding uncertainty interval as a shaded area (Optional)
            # ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='blue', alpha=0.2)
    
            # Adding labels and title
            ax.set_title(f'Line Plot: {target} Over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel(f'Predicted {target}')

            #  # Add gridlines
            # ax.grid(True)  # Add default gridlines
            #  # Optionally, customize the gridlines:
            # ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
            # plt.grid(True)
            plt.legend()
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.fill_between(forecast['ds'], 0, forecast['yhat'], color='lightblue', alpha=0.5)
            ax.plot(forecast['ds'], forecast['yhat'], label=f'Predicted {target}', color='blue')
            ax.set_title(f'Area Plot: {target} Over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel(f'Predicted {target}')

            #  # Add gridlines
            # ax.grid(True)  # Add default gridlines
            #  # Optionally, customize the gridlines:
            # ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
            # plt.grid(True)
            plt.legend()
        
        st.pyplot(fig)

        # Rename columns back to their original names for display and export (after plotting)
        forecast.rename(columns={'ds': date_column, 'yhat': f'Predicted {target}'}, inplace=True)

        # Store predictions in dictionary for later export
        forecast_data[target] = forecast[[date_column, f'Predicted {target}']].tail(prediction_days)

        # Step 14: Display forecasted data without yhat_lower and yhat_upper
        st.write(f"### Forecast for {target} for the next {forecast_horizon_value} {forecast_horizon_unit.lower()}(s)")
        st.dataframe(forecast_data[target])

    # Step 15: Merge all forecasted DataFrames into one, with error handling
    if target_columns:
        try:
            merged_forecasts = forecast_data[target_columns[0]].copy()  # Start with the first target's DataFrame
            for target in target_columns[1:]:
                merged_forecasts = pd.merge(merged_forecasts, forecast_data[target], on=date_column, how='inner')

            st.write("### Forecasts for All Targets")
            st.dataframe(merged_forecasts)

            # Excel file export option with plot type selection
            plot_type_for_excel = st.radio("Choose plot type for Excel download:", ('Line Plot', 'Area Plot'))

            # Step 16: Prepare Excel output with forecast and user-friendly columns
            def to_excel_with_dashboard(merged_forecasts, plot_type_for_excel):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    # Write the merged predictions to an Excel sheet
                    merged_forecasts.to_excel(writer, index=False, sheet_name='Merged_Predictions')

                    # Optionally, add charts or other details
                    workbook = writer.book
                    worksheet = writer.sheets['Merged_Predictions']

                    # Create a chart for predictions for each target
                    for target in target_columns:
                        chart_type = 'line' if plot_type_for_excel == 'Line Plot' else 'area'

                        chart = workbook.add_chart({'type': chart_type})
                        chart.add_series({
                            'name': f'Predicted {target}',
                            'categories': ['Merged_Predictions', 1, 0, len(merged_forecasts), 0],  # Date column
                            'values': ['Merged_Predictions', 1, merged_forecasts.columns.get_loc(f'Predicted {target}'), len(merged_forecasts), merged_forecasts.columns.get_loc(f'Predicted {target}')],  # yhat column
                        })
                        chart.set_x_axis({'name': 'Date'})
                        chart.set_y_axis({'name': f'Predicted {target}'})
                        chart.set_title({'name': f'Predicted {target} Over Time'})

                        worksheet.insert_chart(f'H{2 + 16 * target_columns.index(target)}', chart)

                output.seek(0)
                return output.getvalue()

            # Prepare the merged forecast data for Excel export
            excel_data = to_excel_with_dashboard(merged_forecasts, plot_type_for_excel)

            # Step 17: Provide a download button for the Excel file
            st.download_button(
                label="Download Merged Predictions with Charts",
                data=excel_data,
                file_name='merged_predictions_with_dashboard.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

        except IndexError as e:
            st.error(f"An error occurred while merging forecasts: {str(e)}. Please select at least one target column to predict.")
    else:
        st.error("Please select at least one target column for forecasting.")




