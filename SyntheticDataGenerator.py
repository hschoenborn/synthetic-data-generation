# Standard library imports
from datetime import datetime
from pathlib import Path
import pickle

# Third-party library imports
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from typing import List, Optional
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer, CTGANSynthesizer
from sdv.sequential import PARSynthesizer

class SyntheticDataGenerator:
    def __init__(self, metadata_dict: dict, model_type: str, epochs: int = 300, batch_size: int = 16, cuda: bool = True,
                 enforce_rounding: bool = True, enforce_min_max_values: bool = False,
                 selected_columns: Optional[List[str]] = None):
        """
        Initialize the Synthetic Data Generator with model configuration and metadata.
        """
        self.metadata = self.create_metadata(metadata_dict, selected_columns)
        self.selected_columns = selected_columns
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.cuda = cuda
        self.enforce_rounding = enforce_rounding
        self.enforce_min_max_values = enforce_min_max_values
        self.model = None
        self.losses_df = None

    def create_metadata(self, metadata_dict: dict, selected_columns: Optional[List[str]] = None) -> SingleTableMetadata:
        """
        Filter metadata to include only selected columns if provided.
        """
        if selected_columns is None:
            return SingleTableMetadata().load_from_dict(metadata_dict)

        filtered_metadata = {
            "METADATA_SPEC_VERSION": metadata_dict["METADATA_SPEC_VERSION"],
            "sequence_key": "Measurement ID",
            "sequence_index": "Time",
            "columns": {k: v for k, v in metadata_dict["columns"].items() if k in selected_columns}
        }
        return SingleTableMetadata().load_from_dict(filtered_metadata)

    def preprocess_data(self, real_data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess timestamps by converting milliseconds to human-readable format.
        """
        real_data_df['Time'] = real_data_df['Time'].apply(
            lambda x: datetime.utcfromtimestamp(x / 1000).strftime('%Y-%m-%d %H:%M:%S'))
        return real_data_df

    def postprocess_timestamps(self, df, column='Time'):
        """
        Convert timestamps back to milliseconds for synthetic data.
        """
        df[column] = df[column].apply(lambda x: int(datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp() * 1000))
        return df

    def initialize_model(self):
        """
        Initialize the selected model (TVAE or CTGAN) based on the configuration.
        """
        if self.model_type == "tvae":
            self.model = TVAESynthesizer(
                metadata=self.metadata,
                enforce_rounding=self.enforce_rounding,
                enforce_min_max_values=self.enforce_min_max_values,
                epochs=self.epochs,
                cuda=self.cuda,
            )
        elif self.model_type == "ctgan":
            self.model = CTGANSynthesizer(
                metadata=self.metadata,
                enforce_rounding=self.enforce_rounding,
                enforce_min_max_values=self.enforce_min_max_values,
                epochs=self.epochs,
                cuda=self.cuda,
            )
        elif self.model_type == "par":
            self.model = PARSynthesizer(
                metadata=self.metadata,
                enforce_rounding=self.enforce_rounding,
                enforce_min_max_values=self.enforce_min_max_values,
                epochs=self.epochs,
                cuda=self.cuda,
            )
            all_constraints = list()
            for col in self.selected_columns:
                if col not in ["Time", "Measurement ID"]:
                    if "octet" in col:
                        constraint = {
                            'constraint_class': 'ScalarRange',
                            'constraint_parameters': {
                                'column_name': col,
                                'low_value': 0,
                                'high_value': 100,
                                'strict_boundaries': False
                            }
                        }
                    else:
                        constraint = {
                            'constraint_class': 'Positive',
                            'constraint_parameters': {
                                'column_name': col,
                                'strict_boundaries': False
                            }
                        }
                    all_constraints.append(constraint)
            self.model.add_constraints(constraints=all_constraints)
        else:
            raise ValueError("Unsupported model type")

    def train_model(self, real_data_df: pd.DataFrame):
        """
        Train the model using the real data.
        """
        if self.model is None:
            self.initialize_model()
        self.model.fit(real_data_df)
        if self.model_type != 'par':
            self.losses_df = self.model.get_loss_values()

    def save_model(self, model_folder: Path, model_number: int = 1):
        """
        Save the trained model to a file, ensuring unique filenames.
        """
        model_base_name = f"_{self.model_type}_model_ep{self.epochs}"
        model_name = f'mdl{model_number}{model_base_name}.pkl'
        model_path = model_folder / model_name
        while model_path.exists():
            model_number += 1
            model_name = f'mdl{model_number}{model_base_name}.pkl'
            model_path = model_folder / model_name

        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        return model_path

    def load_model(self, model_path: Path):
        """
        Load a previously saved model.
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def generate_synthetic_data_simple(self, num_samples: int) -> pd.DataFrame:
        """
        Generate synthetic data using the trained simple table model (CTGAN, TVAE).
        """
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        synthetic_data: pd.DataFrame = self.model.sample(num_samples)
        synthetic_data.sort_values(by="Time", inplace=True)
        return synthetic_data

    def generate_synthetic_data_par(self, num_sequences: int) -> pd.DataFrame:
        """
        Generate synthetic data using the trained sequential table model (PAR).
        """
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        synthetic_data = self.model.sample(num_sequences=num_sequences)
        # PAR uses only one measurement ID (sequence=1) in our use case

        return synthetic_data

    def save_synthetic_data(self, synthetic_data: pd.DataFrame, output_file: str):
        """
        Save the generated synthetic data to a CSV file.
        """
        synthetic_data.to_csv(output_file, index=False)

    def plot_graph(self, losses_df: pd.DataFrame, ys: List[str]) -> go.Figure:
        """
        Plot the training loss for the model.
        """
        data = [go.Scatter(x=losses_df['Epoch'], y=losses_df[y], name=y) for y in ys]
        fig = go.Figure(data=data)

        fig.update_layout(template='plotly_white',
                          legend_orientation='h',
                          legend=dict(x=0, y=1.1))

        title = f"{self.model_type.upper()} loss function: "
        fig.update_layout(title=title, xaxis_title='Epoch', yaxis_title='Loss')
        fig.show()

        return fig

    def visualize_loss_values(self) -> go.Figure:
        """
        Visualize loss values for the model training process.
        """
        if self.losses_df is None:
            raise ValueError("Loss values are not available. Train the model first.")

        if self.model_type == "ctgan":
            columns = ["Discriminator Loss", "Generator Loss"]
        elif self.model_type == "tvae":
            columns = ["Loss"]
        else:
            raise ValueError("Unsupported model type for visualization")

        return self.plot_graph(self.losses_df, columns)

    def run_diagnostics(self, real_data_df: pd.DataFrame, synthetic_data: pd.DataFrame):
        """
        Run diagnostic checks on the synthetic data.
        """
        diagnostic_report = run_diagnostic(
            real_data=real_data_df,
            synthetic_data=synthetic_data,
            metadata=self.metadata
        )
        return diagnostic_report

    def evaluate_quality(self, real_data_df: pd.DataFrame, synthetic_data: pd.DataFrame):
        """
        Evaluate the quality of the generated synthetic data.
        """
        quality_report = evaluate_quality(
            real_data=real_data_df,
            synthetic_data=synthetic_data,
            metadata=self.metadata
        )
        return quality_report

    def plot_column_distributions(self, real_data_df: pd.DataFrame, synthetic_data: pd.DataFrame):
        """
        Evaluate the column distributions of the generated synthetic data in comparison to the real data.
        """
        # Define the number of columns in the DataFrames
        num_columns = len(real_data_df.columns)

        # Create subplots: num_columns rows and 2 columns
        fig, axes = plt.subplots(nrows=num_columns, ncols=2, figsize=(12, 3 * num_columns), constrained_layout=True)

        # Iterate through columns and plot
        for i, column in enumerate(real_data_df.columns):
            # Plot real data distribution
            sns.histplot(real_data_df[column], ax=axes[i, 0], kde=True, bins=30)
            axes[i, 0].set_title(f'Real Data: {column}')

            # Plot synthetic data distribution
            sns.histplot(synthetic_data[column], ax=axes[i, 1], kde=True, bins=30)
            axes[i, 1].set_title(f'Synthetic Data: {column}')

            # Set the same x and y limits and ticks for both plots
            x_min = min(real_data_df[column].min(), synthetic_data[column].min())
            x_max = max(real_data_df[column].max(), synthetic_data[column].max())
            y_min = 0
            y_max = max(axes[i, 0].get_ylim()[1], axes[i, 1].get_ylim()[1])

            axes[i, 0].set_xlim(x_min, x_max)
            axes[i, 1].set_xlim(x_min, x_max)
            axes[i, 0].set_ylim(y_min, y_max)
            axes[i, 1].set_ylim(y_min, y_max)

            # Set the same x and y ticks
            x_ticks = sorted(set(axes[i, 0].get_xticks()).union(set(axes[i, 1].get_xticks())))
            y_ticks = sorted(set(axes[i, 0].get_yticks()).union(set(axes[i, 1].get_yticks())))

            axes[i, 0].set_xticks(x_ticks)
            axes[i, 1].set_xticks(x_ticks)
            axes[i, 0].set_yticks(y_ticks)
            axes[i, 1].set_yticks(y_ticks)

        # Show the plot
        plt.show()

    def plot_time_series_comparison(self, real_data_df: pd.DataFrame, synthetic_data_df: pd.DataFrame,
                                    column: str) -> go.Figure:
        """
        Plot a time series comparison between real and synthetic data for a selected column.
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=real_data_df['Time'], y=real_data_df[column], mode='lines', name='Real Data'))
        fig.add_trace(
            go.Scatter(x=synthetic_data_df['Time'], y=synthetic_data_df[column], mode='lines', name='Synthetic Data'))

        fig.update_layout(
            title=f'Time Series Comparison for {column}',
            xaxis_title='Time',
            yaxis_title=column,
            template='plotly_white'
        )

        return fig
