# Standard library imports
from datetime import datetime
import json

# Third-party library imports
import pandas as pd
import streamlit as st
from SyntheticDataGenerator import SyntheticDataGenerator
from database import SessionLocal, create_tables, SyntheticDataFile
from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata
from value_triple_mapping import ValueTripleMapping

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
    st.info(f"Selected columns: {selected_columns}")
    retval = {
        "METADATA_SPEC_VERSION": metadata_dict["METADATA_SPEC_VERSION"],
        "sequence_key": metadata_dict["sequence_key"],
        "sequence_index": metadata_dict["sequence_index"],
        "columns": {k: v for k, v in metadata_dict["columns"].items() if k in selected_columns}
    }
    # st.info(retval)  # DEBUG
    return retval


def get_selected_columns(value_triples, mapping, slim_columns_only):
    selected_columns_for_generation = ["Measurement ID", "Time", "Interval"]
    # selected_columns_for_display = ["Time", "Interval"]

    for triple in value_triples:
        if slim_columns_only:
            selected_columns_for_generation.append(mapping[triple][2])
            # selected_columns_for_display.append(mapping[triple][2])
        else:
            selected_columns_for_generation.extend(mapping[triple])
            # selected_columns_for_display.extend(mapping[triple])

    return selected_columns_for_generation  #, selected_columns_for_display


def upload_and_process_file(uploaded_file, column='Time'):
    real_data_df = pd.read_csv(uploaded_file, delimiter=';')
    st.info("You must provide the Measurement ID (sequence) of your dataset", icon='⚠️')
    measurement_id = st.text_input("Measurement ID:")
    real_data_df['Measurement ID'] = measurement_id
    columns = ['Measurement ID'] + [col for col in real_data_df if col != 'Measurement ID']
    real_data_df = real_data_df[columns]

    return preprocess_timestamps(real_data_df, column=column)

def generate_download_button_from_db(file_id):
    db = SessionLocal()
    file_record = db.query(SyntheticDataFile).filter(SyntheticDataFile.id == file_id).first()

    if file_record:
        return st.download_button(
            label="Download Synthetic Data as CSV",
            data=file_record.content,
            file_name=file_record.filename,
            mime='text/csv'
        )
    else:
        st.error("File not found in the database.")

# def generate_downloadable_csv(dataframe):
#     csv = dataframe.to_csv(index=False).encode('utf-8')
#     return st.download_button(
#         label="Download Synthetic Data as CSV",
#         data=csv,
#         file_name=f"synthetic_data_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv",
#         mime='text/csv'
#     )


def generate_data_app():
    st.title("Synthetic Data Generator")

    # Add a header image
    # st.image("logos/uni_logo.jpg", width=200)

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

        # non_generation_columns = ["Time", "Interval"]
        selected_columns_for_generation = get_selected_columns(
            value_triples, value_triple_mapping, slim_columns_only
        )

        # selected_columns_for_generation, selected_columns_for_display = get_selected_columns(
        #     value_triples, value_triple_mapping, slim_columns_only
        # )
        #
        # # TODO: drop measurement id column in time-series comparisons
        #
        real_data_df = real_data_df[selected_columns_for_generation]  # selected_columns_for_display
        metadata = SingleTableMetadata().load_from_dict(filter_metadata(metadata_dict, selected_columns_for_generation))

        # Model settings
        st.subheader("Step 3: Select Model and Training Settings")
        model_type = st.selectbox("Select model type", ["par", "tvae", "ctgan"])
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
                    st.session_state['generator'] = generator
                    st.success("Model trained successfully.")
                    if model_type != "par":
                        fig = generator.visualize_loss_values()
                        st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"Error during training: {e}")
                    st.error(metadata.to_dict())  # TODO: Delete

        # Data generation
        if 'generator' in st.session_state:
            st.subheader("Step 4: Generate and Download Synthetic Data")

            if st.button("Generate Synthetic Data"):
                with st.spinner('Generating synthetic data...'):
                    try:
                        if model_type != "par":
                            num_samples = len(real_data_df)
                            synthetic_data = st.session_state['generator'].generate_synthetic_data_simple(num_samples)
                        else:
                            num_sequences = int(real_data_df['Measurement ID'].nunique())  # 1 sequence per ID
                            synthetic_data = st.session_state['generator'].generate_synthetic_data_par(num_sequences)

                        synthetic_data_with_time = synthetic_data
                        # pd.concat([real_data_df[non_generation_columns], synthetic_data], axis=1)

                        # Store synthetic data in session state for later use
                        st.session_state['synthetic_data'] = synthetic_data
                        st.session_state['synthetic_data_with_time'] = synthetic_data_with_time

                        st.write("Synthetic Data Sample:")
                        st.dataframe(synthetic_data_with_time)
                        st.success("Synthetic data generated successfully.")
                    except Exception as e:
                        st.error(f"Error during data generation: {e}")

        # Save and evaluate options
        if 'synthetic_data_with_time' in st.session_state:
            st.subheader("Step 5: Save or Evaluate Synthetic Data")

            if st.button("Save Synthetic Data to Database without ID column"):
                with st.spinner('Saving synthetic data to database...'):
                    try:
                        # Drop 'Measurement ID' column
                        synthetic_data = st.session_state['synthetic_data_with_time'].copy().drop(columns=['Measurement ID'], axis=1)
                        # Postprocess the timestamps back to unix format
                        processed_data = st.session_state['generator'].postprocess_timestamps(synthetic_data)

                        # Initialize the database session to save CSV
                        db = SessionLocal()
                        csv_bytes = processed_data.to_csv(index=False).encode('utf-8')

                        new_file = SyntheticDataFile(
                            filename=f"synthetic_data_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv",
                            content=csv_bytes
                        )
                        db.add(new_file)
                        db.commit()
                        db.refresh(new_file)

                        st.success(f"Data saved to database with filename: {new_file.filename}")

                        # Generate download button from the database
                        generate_download_button_from_db(new_file.id)

                    except Exception as e:
                        st.error(f"Error saving data to database: {e}")

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

            if st.button("Plot Distributions of Real vs. Synthetic Data"):
                with st.spinner("Plotting..."):
                    try:
                        generator = st.session_state['generator']
                        fig = generator.plot_column_distributions(real_data_df, st.session_state['synthetic_data'])
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error during time series comparison: {e}")

            selected_column = st.selectbox("Select a column for time series comparison",
                                           selected_columns_for_generation)
            if st.button("Plot Time Series Comparison"):
                with st.spinner("Plotting time series comparison..."):
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
