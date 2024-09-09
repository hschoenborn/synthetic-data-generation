import streamlit as st
import pandas as pd
import json
from datetime import datetime
from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata
from SyntheticDataGenerator import SyntheticDataGenerator
from database import SessionLocal, create_tables, SyntheticDataFile
from sqlalchemy.orm import Session
from value_triple_mapping import ValueTripleMapping
import os

# Initialize the database
create_tables()

# Load metadata
with open('metadata_dict_measurements.json', 'r') as f:
    metadata_dict = json.load(f)

# Initialize ValueTripleMapping
value_triple_mapping = ValueTripleMapping().get_mapping()

# Utility functions
def preprocess_timestamps(df: pd.DataFrame, column: str = 'Time') -> pd.DataFrame:
    df[column] = df[column].apply(
        lambda x: datetime.utcfromtimestamp(x / 1000).strftime('%Y-%m-%d %H:%M:%S')
    )
    return df


def filter_metadata(metadata_dict, selected_columns):
    return {
        "METADATA_SPEC_VERSION": metadata_dict["METADATA_SPEC_VERSION"],
        "columns": {k: v for k, v in metadata_dict["columns"].items() if k in selected_columns}
    }


def get_selected_columns(value_triples, mapping, slim_columns_only):
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
    real_data_df = pd.read_csv(uploaded_file, delimiter=';')
    return preprocess_timestamps(real_data_df, column=column)


def generate_downloadable_csv(dataframe):
    csv = dataframe.to_csv(index=False).encode('utf-8')
    return st.download_button(
        label="Download Synthetic Data as CSV",
        data=csv,
        file_name='synthetic_data.csv',
        mime='text/csv'
    )


def generate_data_app():
    st.title("Synthetic Data Generator")

    # Add a header image
    st.image("logos/uni_logo.jpg", width=200)

    # File upload section
    st.subheader("Step 1: Upload Your Data")
    st.write("Please upload a CSV file with the common format of a StableNet measurement to start the process.")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        real_data_df = upload_and_process_file(uploaded_file)
        st.write("Real Data Sample:")
        st.dataframe(real_data_df)

        # Step 2: Configure columns and options
        st.subheader("Step 2: Configure Data Generation Options")
        slim_columns_only = st.checkbox(
            "Generate data for Avg columns only (keeps timestamp and interval for later consistency)"
        )

        value_triples = st.multiselect(
            "Select value-triples to include",
            list(value_triple_mapping.keys()),
            list(value_triple_mapping.keys())  # Default selected
        )

        non_generation_columns = ["Time", "Interval"]
        selected_columns_for_generation, selected_columns_for_display = get_selected_columns(
            value_triples, value_triple_mapping, slim_columns_only
        )

        real_data_df = real_data_df[selected_columns_for_display]
        metadata = SingleTableMetadata().load_from_dict(filter_metadata(metadata_dict, selected_columns_for_generation))

        # Model settings
        st.subheader("Step 3: Select Model and Training Settings")
        model_type = st.selectbox("Select model type", ["tvae", "ctgan"])
        epochs = st.number_input("Enter number of epochs", min_value=1, max_value=1000, value=300)

        # Model training
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

        # Data generation
        if 'generator' in st.session_state:
            st.subheader("Step 4: Generate and Download Synthetic Data")
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
                        st.success("Synthetic data generated successfully.")
                    except Exception as e:
                        st.error(f"Error during data generation: {e}")

        # Save and evaluate options
        if 'synthetic_data' in st.session_state:
            st.subheader("Step 5: Save or Evaluate Synthetic Data")

            if st.button("Evaluate Synthetic Data"):
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

            if st.button("Save Synthetic Data to Database"):
                with st.spinner('Saving synthetic data to database...'):
                    try:
                        generate_downloadable_csv(
                            st.session_state['generator'].postprocess_timestamps(synthetic_data_with_time.copy())
                        )
                        synthetic_data = st.session_state['synthetic_data']
                        csv_bytes = synthetic_data.to_csv(index=False).encode('utf-8')
                        db = SessionLocal()
                        new_file = SyntheticDataFile(
                            filename=f"synthetic_data_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv",
                            content=csv_bytes
                        )
                        db.add(new_file)
                        db.commit()
                        db.refresh(new_file)
                        st.success(f"Data saved to database with filename: {new_file.filename}")
                    except Exception as e:
                        st.error(f"Error saving data to database: {e}")

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


def data_app():
    st.header("Stored Synthetic Data Files")
    st.image("logos/Infosim_logo.png", width=200)

    db = SessionLocal()
    files = db.query(SyntheticDataFile).order_by(SyntheticDataFile.created_at.desc()).all()

    if files:
        for file in files:
            st.subheader(f"Filename: {file.filename}")
            st.write(f"Uploaded at: {file.created_at}")

            st.download_button(
                label="Download CSV",
                data=file.content,
                file_name=file.filename,
                mime='text/csv'
            )
    else:
        st.info("No files found in the database.")

def main():

    # Create placeholders for top and bottom logos
    top_logo = st.sidebar.empty()

    # Display the top logo
    top_logo.image("logos/Infosim_logo.png", use_column_width=True)

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Generate Data", "Data"])

    if app_mode == "Generate Data":
        generate_data_app()
    elif app_mode == "Data":
        data_app()

    # Add an empty element to force space before the bottom logo
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.write("\n")

    # Display the bottom logo in a placeholder at the bottom
    bottom_logo = st.sidebar.empty()
    bottom_logo.image("logos/uni_logo.jpg", use_column_width=True)


if __name__ == "__main__":
    main()


