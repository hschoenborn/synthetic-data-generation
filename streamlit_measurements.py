import streamlit as st
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from SyntheticDataGenerator import SyntheticDataGenerator
from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata

# Load metadata
with open('metadata_dict_measurements.json', 'r') as f:
    metadata_dict = json.load(f)

# Define the mapping from user-friendly names to actual column names
value_triple_mapping = {
    "op-reachability": ["Min op-reachability (%)", "Max op-reachability (%)", "Avg op-reachability (%)"],
    "in-octets": ["Min in-octets (kbit/s)", "Max in-octets (kbit/s)", "Avg in-octets (kbit/s)"],
    "in-utilization": ["Min in-utilization (%)", "Max in-utilization (%)", "Avg in-utilization (%)"],
    "in-errors": ["Min in-errors (%)", "Max in-errors (%)", "Avg in-errors (%)"],
    "in-discards": ["Min in-discards (%)", "Max in-discards (%)", "Avg in-discards (%)"],
    "out-octets": ["Min out-octets (kbit/s)", "Max out-octets (kbit/s)", "Avg out-octets (kbit/s)"],
    "out-utilization": ["Min out-utilization (%)", "Max out-utilization (%)", "Avg out-utilization (%)"],
    "out-errors": ["Min out-errors (%)", "Max out-errors (%)", "Avg out-errors (%)"],
    "out-discards": ["Min out-discards (%)", "Max out-discards (%)", "Avg out-discards (%)"]
}


def main():
    st.title("Synthetic Data Generator")

    # File upload
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        real_data_df = pd.read_csv(uploaded_file, delimiter=';')
        st.write("Real Data Sample:")
        st.dataframe(real_data_df)

        # Select value-triples
        deselect_all = st.checkbox("Deselect all value-triples")

        if deselect_all:
            value_triples = []
        else:
            value_triples = st.multiselect(
                "Select value-triples to include",
                list(value_triple_mapping.keys()),
                list(value_triple_mapping.keys())  # default selected
            )

        selected_columns = ["Time", "Interval"]
        for triple in value_triples:
            selected_columns.extend(value_triple_mapping[triple])

        real_data_df = real_data_df[selected_columns]

        # Update metadata based on selected columns
        filtered_metadata = {
            "METADATA_SPEC_VERSION": metadata_dict["METADATA_SPEC_VERSION"],
            "columns": {k: v for k, v in metadata_dict["columns"].items() if k in selected_columns}
        }
        metadata = SingleTableMetadata().load_from_dict(filtered_metadata)

        # Model selection
        model_type = st.selectbox("Select model type", ["tvae", "ctgan"])

        # Epochs input
        epochs = st.number_input("Enter number of epochs", min_value=1, max_value=1000, value=300)

        if st.button("Train Model"):
            with st.spinner('Training model...'):
                try:
                    generator = SyntheticDataGenerator(metadata_dict, model_type=model_type, epochs=epochs,
                                                       selected_columns=selected_columns)
                    real_data_df = generator.preprocess_data(real_data_df)
                    generator.train_model(real_data_df)

                    # Plot the loss values
                    fig = generator.visualize_loss_values()
                    st.plotly_chart(fig)
                    st.session_state['generator'] = generator  # Store the generator in session state
                    st.success("Model trained successfully.")
                except Exception as e:
                    st.error(f"Error during training: {e}")

        # Generate synthetic data
        if 'generator' in st.session_state:
            num_samples = st.number_input("Number of synthetic rows to generate", min_value=1, max_value=10000,
                                          value=500)
            if st.button("Generate Synthetic Data"):
                with st.spinner('Generating synthetic data...'):
                    try:
                        synthetic_data = st.session_state['generator'].generate_synthetic_data(num_samples)
                        st.write("Synthetic Data Sample:")
                        st.dataframe(synthetic_data)

                        # Store synthetic data in session state
                        st.session_state['synthetic_data'] = synthetic_data

                        # Convert timestamps back before downloading
                        synthetic_data_for_download = st.session_state['generator'].postprocess_timestamps(
                            synthetic_data.copy())

                        # Download synthetic data
                        csv = synthetic_data_for_download.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Synthetic Data as CSV",
                            data=csv,
                            file_name='synthetic_data.csv',
                            mime='text/csv',
                        )
                        st.success("Synthetic data generated successfully.")
                    except Exception as e:
                        st.error(f"Error during data generation: {e}")

            # Evaluate synthetic data quality
            if 'synthetic_data' in st.session_state and st.button("Evaluate Synthetic Data"):
                with st.spinner('Evaluating synthetic data...'):
                    try:
                        report = QualityReport()
                        report.generate(real_data_df, st.session_state['synthetic_data'], metadata.to_dict())

                        overall_score = report.get_score()
                        st.write(f"Overall Score: {overall_score:.2f}")

                        properties_df = report.get_properties()
                        st.write("Properties and Scores:")
                        st.write(properties_df)

                        for property_name in properties_df['Property']:
                            st.write(f"Details for {property_name}:")
                            details_df = report.get_details(property_name)
                            st.write(details_df)
                            fig = report.get_visualization(property_name)
                            st.plotly_chart(fig)

                        st.success("Evaluation completed successfully.")
                    except Exception as e:
                        st.error(f"Error during evaluation: {e}")

            # Time series comparison
            selected_column = st.selectbox("Select a column for time series comparison", selected_columns[2:])
            if 'synthetic_data' in st.session_state and st.button("Plot Time Series Comparison"):
                with st.spinner('Plotting time series comparison...'):
                    try:
                        fig = st.session_state['generator'].plot_time_series_comparison(real_data_df, st.session_state[
                            'synthetic_data'], selected_column)
                        st.plotly_chart(fig)
                    except Exception as e:
                        st.error(f"Error during time series comparison: {e}")


if __name__ == "__main__":
    main()
