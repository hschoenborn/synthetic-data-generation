import streamlit as st
import pandas as pd
import json
from datetime import datetime
from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata
from SyntheticDataGenerator import SyntheticDataGenerator


# Utility functions
def preprocess_timestamps(df: pd.DataFrame, column: str = 'Time') -> pd.DataFrame:
    """
    Convert timestamps from milliseconds to human-readable format.
    """
    df[column] = df[column].apply(
        lambda x: datetime.utcfromtimestamp(x / 1000).strftime('%Y-%m-%d %H:%M:%S')
    )
    return df


def filter_metadata(metadata_dict, selected_columns):
    """
    Filter metadata to include only the specified columns.
    """
    return {
        "METADATA_SPEC_VERSION": metadata_dict["METADATA_SPEC_VERSION"],
        "columns": {k: v for k, v in metadata_dict["columns"].items() if k in selected_columns}
    }


def get_selected_columns(value_triples, mapping, slim_columns_only):
    """
    Generate lists of selected columns for data generation and display.
    """
    selected_columns_for_generation = []
    selected_columns_for_display = ["Time", "Interval"]

    for triple in value_triples:
        if slim_columns_only:
            selected_columns_for_generation.append(mapping[triple][2])
            selected_columns_for_display.append(mapping[triple][2])
        else:
            selected_columns_for_generation.extend(mapping[triple])
            selected_columns_for_display.extend(mapping[triple])

    return selected_columns_for_generation, selected_columns_for_display


def upload_and_process_file(uploaded_file, column='Time'):
    """
    Handle file upload and timestamp processing.
    """
    real_data_df = pd.read_csv(uploaded_file, delimiter=';')
    return preprocess_timestamps(real_data_df, column=column)


def generate_downloadable_csv(dataframe):
    """
    Create a downloadable CSV file for synthetic data.
    """
    csv = dataframe.to_csv(index=False).encode('utf-8')
    return st.download_button(
        label="Download Synthetic Data as CSV",
        data=csv,
        file_name='synthetic_data.csv',
        mime='text/csv'
    )


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
        real_data_df = upload_and_process_file(uploaded_file)
        st.write("Real Data Sample:")
        st.dataframe(real_data_df)

        slim_columns_only = st.checkbox(
            "Generate data for Avg columns only (keeps timestamp and interval for later consistency)")

        deselect_all = st.checkbox("Deselect all value-triples")
        value_triples = [] if deselect_all else st.multiselect(
            "Select value-triples to include",
            list(value_triple_mapping.keys()),
            list(value_triple_mapping.keys())  # Default selected
        )

        non_generation_columns = ["Time", "Interval"]
        selected_columns_for_generation, selected_columns_for_display = get_selected_columns(
            value_triples, value_triple_mapping, slim_columns_only)

        real_data_df = real_data_df[selected_columns_for_display]
        metadata = SingleTableMetadata().load_from_dict(filter_metadata(metadata_dict, selected_columns_for_generation))

        model_type = st.selectbox("Select model type", ["tvae", "ctgan"])
        epochs = st.number_input("Enter number of epochs", min_value=1, max_value=1000, value=300)

        if st.button("Train Model"):
            with st.spinner('Training model...'):
                try:
                    generator = SyntheticDataGenerator(
                        metadata_dict, model_type=model_type, epochs=epochs,
                        selected_columns=selected_columns_for_generation
                    )
                    generator.train_model(real_data_df[selected_columns_for_generation])
                    fig = generator.visualize_loss_values()
                    st.plotly_chart(fig)
                    st.session_state['generator'] = generator
                    st.success("Model trained successfully.")
                except Exception as e:
                    st.error(f"Error during training: {e}")

        if 'generator' in st.session_state:
            num_samples = len(real_data_df)
            if st.button("Generate Synthetic Data"):
                with st.spinner('Generating synthetic data...'):
                    try:
                        synthetic_data = st.session_state['generator'].generate_synthetic_data(num_samples)
                        synthetic_data_with_time = pd.concat([real_data_df[non_generation_columns], synthetic_data],
                                                             axis=1)
                        st.write("Synthetic Data Sample:")
                        st.dataframe(synthetic_data_with_time)
                        st.session_state['synthetic_data'] = synthetic_data_with_time
                        generate_downloadable_csv(
                            st.session_state['generator'].postprocess_timestamps(synthetic_data_with_time.copy()))
                        st.success("Synthetic data generated successfully.")
                    except Exception as e:
                        st.error(f"Error during data generation: {e}")

            if 'synthetic_data' in st.session_state and st.button("Evaluate Synthetic Data"):
                with st.spinner('Evaluating synthetic data...'):
                    try:
                        generator = st.session_state['generator']
                        report = QualityReport()
                        report.generate(real_data_df[selected_columns_for_generation],
                                        st.session_state['synthetic_data'][selected_columns_for_generation],
                                        metadata.to_dict())
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

            selected_column = st.selectbox("Select a column for time series comparison",
                                           selected_columns_for_generation)
            if st.button("Plot Time Series Comparison"):
                with st.spinner('Plotting time series comparison...'):
                    try:
                        generator = st.session_state['generator']
                        fig = generator.plot_time_series_comparison(real_data_df, st.session_state['synthetic_data'],
                                                                    selected_column)
                        st.plotly_chart(fig)
                    except Exception as e:
                        st.error(f"Error during time series comparison: {e}")


if __name__ == "__main__":
    main()
