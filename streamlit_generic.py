import streamlit_generic as st
import pandas as pd
from DataAccessor import DataAccessor  # Ensure the DataAccessor class is in a file named data_accessor.py

def main():
    st.title("CSV Data Viewer and Editor")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Assume the CSV is semicolon-separated based on the example provided
        accessor = DataAccessor(uploaded_file, separator=';')

        st.subheader("Data Overview")
        st.write(accessor.df)

        st.subheader("Define Timestamp and Interval Columns")
        timestamp_column = st.selectbox("Select the timestamp column", accessor.df.columns)
        interval_column = st.selectbox("Select the interval column", accessor.df.columns)

        if st.button("Set Timestamp Column"):
            accessor.set_timestamp_column(timestamp_column)
            st.success(f"Set {timestamp_column} as the timestamp column")
            st.write(accessor.df)

        st.subheader("Column Information")
        st.dataframe(accessor.overview())

        st.subheader("Adjust Column Data Types")
        column_to_adjust = st.selectbox("Select column to adjust", accessor.df.columns)
        new_type = st.selectbox("Select new data type", ["datetime", "numeric", "string"])

        if st.button("Update Column Type"):
            accessor.update_column_type(column_to_adjust, new_type)
            st.success(f"Updated {column_to_adjust} to {new_type}")
            st.write(accessor.df)
            st.dataframe(accessor.overview())

if __name__ == "__main__":
    main()
