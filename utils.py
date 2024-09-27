import pandas as pd

def detect_numeric_columns(df):
        """Ensure that numeric columns are properly detected."""
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_columns:  # If no numeric columns found, force conversion
            df = df.apply(pd.to_numeric, errors='coerce')
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        return numeric_columns

      