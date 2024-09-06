import json

import pandas as pd
from datetime import datetime
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer, CTGANSynthesizer
import pickle
from pathlib import Path

def load_real_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, delimiter=';')
    df['Time'] = df['Time'].apply(lambda x: datetime.utcfromtimestamp(x / 1000).strftime('%Y-%m-%d %H:%M:%S'))
    return df

def load_metadata(metadata_dict: dict) -> SingleTableMetadata:
    metadata = SingleTableMetadata().load_from_dict(metadata_dict)
    return metadata

def initialize_model(metadata: SingleTableMetadata, model_type: str, epochs: int, cuda: bool, enforce_rounding: bool, enforce_min_max_values: bool) -> TVAESynthesizer:
    if model_type == "tvae":
        model = TVAESynthesizer(
            metadata=metadata,
            enforce_rounding=enforce_rounding,
            enforce_min_max_values=enforce_min_max_values,
            epochs=epochs,
            cuda=cuda,
        )
    elif model_type == "ctgan":
        model = CTGANSynthesizer(
            metadata=metadata,
            enforce_rounding=enforce_rounding,
            enforce_min_max_values=enforce_min_max_values,
            epochs=epochs,
            cuda=cuda,
        )
    else:
        raise ValueError("Unsupported model type")
    return model

def save_model(model, model_path: Path):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(model_path: Path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def train_model(model, real_data_df: pd.DataFrame):
    model.fit(real_data_df)
    return model

def generate_synthetic_data(model, num_samples: int) -> pd.DataFrame:
    synthetic_data = model.sample(num_samples)
    return synthetic_data

def save_synthetic_data(synthetic_data: pd.DataFrame, output_file: str):
    synthetic_data.to_csv(output_file, index=False)

def main():
    # File paths and configurations
    real_data_file_path = "/content/drive/MyDrive/input/measurements_highest_traffic/rawdata/108503_last_4_6.csv"
    metadata_dict_path = 'metadata_dict_measurements.json'
    model_folder = Path('./models')
    model_folder.mkdir(parents=True, exist_ok=True)
    epochs = 300
    batch_size = 16
    cuda = True
    enforce_rounding = True
    enforce_min_max_values = True
    model_type = "tvae"
    num_samples = 500

    # Load real data
    real_data_df = load_real_data(real_data_file_path)

    # Load metadata
    with open(metadata_dict_path, 'r') as f:
        metadata_dict = json.load(f)
    metadata = load_metadata(metadata_dict)

    # Initialize model
    model = initialize_model(metadata, model_type, epochs, cuda, enforce_rounding, enforce_min_max_values)

    # Train model
    model = train_model(model, real_data_df)

    # Save the trained model
    model_number = 1
    model_base_name = f"_{model_type}_i_108503_last_4_6_rawdata_ep{epochs}"
    model_name = f'mdl{model_number}{model_base_name}.pkl'
    model_path = model_folder / model_name
    while model_path.exists():
        model_number += 1
        model_name = f'mdl{model_number}{model_base_name}.pkl'
        model_path = model_folder / model_name

    save_model(model, model_path)

    # Generate synthetic data
    synthetic_data = generate_synthetic_data(model, num_samples)

    # Save synthetic data
    synthetic_data_output_path = 'synthetic_data.csv'
    save_synthetic_data(synthetic_data, synthetic_data_output_path)

    # Verify the synthetic data
    print(synthetic_data.head())

if __name__ == "__main__":
    main()
