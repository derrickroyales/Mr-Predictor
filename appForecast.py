import streamlit as st
import pandas as pd
from prophet import Prophet
from io import BytesIO

# Title of the web app
st.title("Sales Prediction Web App")

# Step 1: File upload
st.write("### Upload your Excel file:")
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

# Step 2: User inputs for price, average competitor price, and prediction days
custom_price = st.number_input("Enter Your Price:", min_value=0.0, step=0.01)
competitor_price = st.number_input("Enter Competitor Price:", min_value=0.0, step=0.01)
prediction_days = st.number_input("Number of Days for Prediction:", min_value=1, step=1)

# Proceed if a file is uploaded and user inputs are provided
if uploaded_file is not None and custom_price and competitor_price and prediction_days:

    # Step 3: Read Excel data into a Pandas DataFrame
    df = pd.read_excel(uploaded_file)

    # Check if necessary columns are present
    required_columns = ['date', 'sales', 'price', 'average competitor price', 'revenue']
    if not all(col in df.columns for col in required_columns):
        st.error(f"Excel file must contain the following columns: {required_columns}")
    else:
        # Step 4: Rename columns to fit Prophet's expected format
        df.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)

        # Display the uploaded data
        st.write("### Uploaded Data")
        st.dataframe(df.head())

        # Step 5: Train the Prophet model with regressors
        model = Prophet()
        model.add_regressor('price')
        model.add_regressor('average competitor price')

        # Fit the model
        model.fit(df[['ds', 'y', 'price', 'average competitor price']])

        # Step 6: Make predictions for the next N days
        future = model.make_future_dataframe(periods=prediction_days)

        # Add custom user inputs for price and competitor price
        future['price'] = custom_price
        future['average competitor price'] = competitor_price

        # Make predictions
        forecast = model.predict(future)

        # Add 'revenue' column (predicted sales * custom price)
        forecast['revenue'] = custom_price * forecast['yhat']

        # Step 7: Display the forecasted values
        st.write(f"### Forecast for the next {prediction_days} days")
        predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'revenue']].tail(prediction_days)
        st.dataframe(predictions)

        # Step 8: Plot the forecast
        st.write("### Forecast Plot")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        # Step 9: Provide download option for the predictions as an Excel file
        def to_excel_with_dashboard(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Sheet 1: Write the predictions data
                df.to_excel(writer, index=False, sheet_name='Predictions')

                # Access the workbook and worksheet for Sheet 1
                workbook = writer.book
                worksheet1 = writer.sheets['Predictions']

                # Create a chart for predicted sales
                sales_chart = workbook.add_chart({'type': 'line'})
                sales_chart.add_series({
                    'name': 'Predicted Sales',
                    'categories': ['Predictions', 1, 0, len(df), 0],  # Date column
                    'values': ['Predictions', 1, 1, len(df), 1],  # yhat column
                })
                sales_chart.set_x_axis({'name': 'Date'})
                sales_chart.set_y_axis({'name': 'Predicted Sales'})
                sales_chart.set_title({'name': 'Predicted Sales Over Time'})

                # Insert the sales chart in the first worksheet
                worksheet1.insert_chart('H2', sales_chart)

                # Create another chart for revenue predictions
                revenue_chart = workbook.add_chart({'type': 'line'})
                revenue_chart.add_series({
                    'name': 'Predicted Revenue',
                    'categories': ['Predictions', 1, 0, len(df), 0],  # Date column
                    'values': ['Predictions', 1, 4, len(df), 4],  # revenue column
                })
                revenue_chart.set_x_axis({'name': 'Date'})
                revenue_chart.set_y_axis({'name': 'Predicted Revenue'})
                revenue_chart.set_title({'name': 'Predicted Revenue Over Time'})

                # Insert the revenue chart in the first worksheet
                worksheet1.insert_chart('H20', revenue_chart)

                # Move summary data to the 'Predictions' sheet
                worksheet1.write('A20', 'Summary')
                worksheet1.write('A21', 'Average Predicted Sales:')
                worksheet1.write('B21', df['yhat'].mean())

                # Sheet 2: Empty dashboard sheet
                workbook.add_worksheet('Dashboard')

            processed_data = output.getvalue()
            return processed_data

        # Convert the forecast DataFrame to an Excel file with a dashboard
        excel_data = to_excel_with_dashboard(predictions)

        # Step 10: Download button for the Excel file with charts and dashboard
        st.download_button(
            label="Download Predictions with Charts and Dashboard",
            data=excel_data,
            file_name='predictions_with_dashboard.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

