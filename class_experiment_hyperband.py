"""
Experiment class for hyperparameter tuning using Keras Tuner Hyperband.

This class extends the experiment workflow to perform hyperparameter search
using the Hyperband algorithm, then trains the best model and saves results
including optimal hyperparameters.
"""
import os
import json
import sys
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Any, Dict
import keras
import keras_tuner as kt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class experiment_hyperband:
    """
    A class to manage Keras model hyperparameter tuning experiments using Hyperband.

    This class:
    1. Loads model builder function and hyperparameter search space
    2. Uses Hyperband to find optimal hyperparameters
    3. Trains the best model with full epochs
    4. Saves results including optimal hyperparameters and training history
    """

    def __init__(
        self,
        model_name: str,
        experiment_name: str,
        max_epochs: int = 50,
        hyperband_iterations: int = 2,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 1,
        objective: str = 'val_accuracy',
        **kwargs
    ):
        """
        Initialize a hyperparameter tuning experiment.

        Args:
            model_name: Name of the model (used to load builder and hyperparams)
            experiment_name: Name for this experiment (used in result files)
            max_epochs: Maximum number of epochs for Hyperband
            hyperband_iterations: Number of times to iterate over the full Hyperband algorithm
            batch_size: Batch size for training
            validation_split: Fraction of training data to use for validation
            verbose: Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)
            objective: Metric to optimize ('val_accuracy', 'val_loss', etc.)
            **kwargs: Additional parameters to pass to model.fit()
        """
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.max_epochs = max_epochs
        self.hyperband_iterations = hyperband_iterations
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.verbose = verbose
        self.objective = objective
        self.fit_kwargs = kwargs

        self.model_builder = None
        self.hyperparameter_space = None
        self.tuner = None
        self.best_model = None
        self.best_hyperparameters = None
        self.history = None
        self.test_results = None

        # Ensure results directory exists
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)

        # Tuner directory
        self.tuner_dir = Path(__file__).parent / "tuner_logs" / experiment_name
        self.tuner_dir.mkdir(parents=True, exist_ok=True)

        # Load model builder and hyperparameters
        self._load_model_builder()
        self._load_hyperparameters()

    def _load_model_builder(self) -> None:
        """Load the model builder function from the models directory."""
        models_dir = Path(__file__).parent / "models"
        builder_path = models_dir / f"{self.model_name}_builder.py"

        if not builder_path.exists():
            raise FileNotFoundError(
                f"Model builder not found: {builder_path}\n"
                f"Please create it using save_model_hyperband.py"
            )

        # Dynamically import the builder module
        spec = importlib.util.spec_from_file_location(f"{self.model_name}_builder", builder_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{self.model_name}_builder"] = module
        spec.loader.exec_module(module)

        self.model_builder = module.build_model
        print(f"Loaded model builder from: {builder_path}")

    def _load_hyperparameters(self) -> None:
        """Load hyperparameter search space from JSON file."""
        models_dir = Path(__file__).parent / "models"
        hyperparams_path = models_dir / f"{self.model_name}_hyperparams.json"

        if not hyperparams_path.exists():
            raise FileNotFoundError(
                f"Hyperparameters not found: {hyperparams_path}\n"
                f"Please create it using save_model_hyperband.py"
            )

        with open(hyperparams_path, 'r') as f:
            self.hyperparameter_space = json.load(f)

        print(f"Loaded hyperparameter space from: {hyperparams_path}")

    def run(
        self,
        x_train: Any,
        y_train: Any,
        x_test: Any,
        y_test: Any
    ) -> Tuple[keras.callbacks.History, dict, dict]:
        """
        Run the hyperparameter tuning experiment.

        Args:
            x_train: Training features
            y_train: Training labels
            x_test: Test features
            y_test: Test labels

        Returns:
            Tuple of (training history, test metrics, best hyperparameters)
        """
        print(f"\n{'='*70}")
        print(f"HYPERBAND EXPERIMENT: {self.experiment_name}")
        print(f"{'='*70}\n")

        # Step 1: Setup Hyperband tuner
        print("Step 1: Setting up Hyperband tuner...")
        self.tuner = kt.Hyperband(
            self.model_builder,
            objective=self.objective,
            max_epochs=self.max_epochs,
            hyperband_iterations=self.hyperband_iterations,
            directory=str(self.tuner_dir.parent),
            project_name=self.experiment_name,
            overwrite=False  # Set to True to start fresh
        )

        print(f"  Objective: {self.objective}")
        print(f"  Max epochs: {self.max_epochs}")
        print(f"  Hyperband iterations: {self.hyperband_iterations}")
        print(f"  Search space size: {len(self.hyperparameter_space['hyperparameters'])} hyperparameters")

        # Step 2: Search for best hyperparameters
        print(f"\nStep 2: Searching for optimal hyperparameters...")
        print("  This may take a while...\n")

        self.tuner.search(
            x_train,
            y_train,
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=self.verbose,
            **self.fit_kwargs
        )

        # Step 3: Get best hyperparameters
        print("\nStep 3: Retrieving best hyperparameters...")
        self.best_hyperparameters = self.tuner.get_best_hyperparameters(num_trials=1)[0]

        print("\n  Optimal hyperparameters found:")
        for hp_name in self.best_hyperparameters.values.keys():
            print(f"    {hp_name}: {self.best_hyperparameters.get(hp_name)}")

        # Step 4: Train best model with full epochs
        print(f"\nStep 4: Training best model for {self.max_epochs} epochs...")
        self.best_model = self.tuner.get_best_models(num_models=1)[0]

        # Retrain on full data
        self.history = self.best_model.fit(
            x_train,
            y_train,
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=self.verbose,
            **self.fit_kwargs
        )

        # Step 5: Evaluate on test set
        print("\nStep 5: Evaluating on test set...")
        test_metrics = self.best_model.evaluate(x_test, y_test, verbose=self.verbose, return_dict=True)
        self.test_results = test_metrics

        # Step 6: Save results
        print("\nStep 6: Saving results...")
        self._save_results()

        print(f"\n{'='*70}")
        print(f"HYPERBAND EXPERIMENT COMPLETE: {self.experiment_name}")
        print(f"{'='*70}\n")

        return self.history, self.test_results, self.best_hyperparameters.values

    def _save_results(self) -> str:
        """
        Save experiment results including optimal hyperparameters.

        Returns:
            Path to the saved results file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"{self.experiment_name}_{timestamp}.json"
        results_path = self.results_dir / results_filename

        # Prepare results dictionary
        results = {
            "experiment_name": self.experiment_name,
            "model_name": self.model_name,
            "timestamp": timestamp,
            "datetime": datetime.now().isoformat(),
            "tuning_method": "Hyperband",
            "parameters": {
                "max_epochs": self.max_epochs,
                "hyperband_iterations": self.hyperband_iterations,
                "batch_size": self.batch_size,
                "validation_split": self.validation_split,
                "objective": self.objective,
                **self.fit_kwargs
            },
            "optimal_hyperparameters": self.best_hyperparameters.values,
            "hyperparameter_search_space": self.hyperparameter_space,
            "model_summary": self._get_model_summary(),
            "training_history": {
                key: [float(val) for val in values]
                for key, values in self.history.history.items()
            },
            "test_metrics": {
                key: float(value) if isinstance(value, (int, float)) else value
                for key, value in self.test_results.items()
            }
        }

        # Save as JSON
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Save human-readable summary
        summary_path = self.results_dir / f"{self.experiment_name}_{timestamp}_summary.txt"
        self._save_summary(summary_path, results)

        # Save training history plot
        plot_path = self.results_dir / f"{self.experiment_name}_{timestamp}_history.png"
        self._plot_history(plot_path)

        # Save the best model
        model_path = self.results_dir / f"{self.experiment_name}_{timestamp}_best_model.keras"
        self.best_model.save(model_path)

        print(f"\n  Results saved to:")
        print(f"    JSON: {results_path}")
        print(f"    Summary: {summary_path}")
        print(f"    Plot: {plot_path}")
        print(f"    Best Model: {model_path}")

        return str(results_path)

    def _get_model_summary(self) -> dict:
        """Get a dictionary representation of the model architecture."""
        total_params = sum([keras.ops.size(w).numpy() for w in self.best_model.trainable_weights])

        return {
            "total_parameters": int(total_params),
            "layers": len(self.best_model.layers),
            "input_shape": str(self.best_model.input_shape),
            "output_shape": str(self.best_model.output_shape)
        }

    def _save_summary(self, path: Path, results: dict) -> None:
        """Save a human-readable text summary of the experiment."""
        with open(path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write(f"HYPERBAND TUNING SUMMARY: {self.experiment_name}\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Date/Time: {results['datetime']}\n")
            f.write(f"Model: {results['model_name']}\n")
            f.write(f"Tuning Method: {results['tuning_method']}\n\n")

            f.write("TUNING PARAMETERS:\n")
            f.write("-" * 70 + "\n")
            for key, value in results['parameters'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            f.write("OPTIMAL HYPERPARAMETERS:\n")
            f.write("-" * 70 + "\n")
            for key, value in results['optimal_hyperparameters'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            f.write("MODEL ARCHITECTURE (with optimal hyperparameters):\n")
            f.write("-" * 70 + "\n")
            for key, value in results['model_summary'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            f.write("TRAINING RESULTS:\n")
            f.write("-" * 70 + "\n")
            history = results['training_history']
            for metric_name, values in history.items():
                final_value = values[-1]
                f.write(f"  Final {metric_name}: {final_value:.6f}\n")
            f.write("\n")

            f.write("TEST SET EVALUATION:\n")
            f.write("-" * 70 + "\n")
            for metric_name, value in results['test_metrics'].items():
                f.write(f"  {metric_name}: {value:.6f}\n")
            f.write("\n")

            f.write("TRAINING HISTORY (per epoch):\n")
            f.write("-" * 70 + "\n")
            metric_names = list(history.keys())
            header = f"{'Epoch':<8}"
            for name in metric_names:
                header += f"{name:<20}"
            f.write(header + "\n")
            f.write("-" * 70 + "\n")

            num_epochs = len(history[metric_names[0]])
            for epoch in range(num_epochs):
                row = f"{epoch + 1:<8}"
                for name in metric_names:
                    row += f"{history[name][epoch]:<20.6f}"
                f.write(row + "\n")

            f.write("\n" + "=" * 70 + "\n")

    def _plot_history(self, path: Path) -> None:
        """Plot training history and save as PNG."""
        history = self.history.history

        # Separate metrics into training and validation
        train_metrics = {}
        val_metrics = {}

        for key, values in history.items():
            if key.startswith('val_'):
                val_metrics[key[4:]] = values
            else:
                train_metrics[key] = values

        # Determine number of subplots
        metric_names = list(train_metrics.keys())
        num_metrics = len(metric_names)

        if num_metrics == 0:
            return

        # Create figure
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics))
        if num_metrics == 1:
            axes = [axes]

        epochs = range(1, len(train_metrics[metric_names[0]]) + 1)

        for idx, metric_name in enumerate(metric_names):
            ax = axes[idx]

            # Plot training and validation
            ax.plot(epochs, train_metrics[metric_name], 'b-', label=f'Training {metric_name}', linewidth=2)
            if metric_name in val_metrics:
                ax.plot(epochs, val_metrics[metric_name], 'r-', label=f'Validation {metric_name}', linewidth=2)

            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric_name.capitalize(), fontsize=12)
            ax.set_title(f'{metric_name.capitalize()} over Epochs (Hyperband Tuned)', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)

            # Annotations
            if metric_name in val_metrics:
                best_val = max(val_metrics[metric_name]) if 'acc' in metric_name else min(val_metrics[metric_name])
                best_val_epoch = val_metrics[metric_name].index(best_val) + 1
                ax.axhline(y=best_val, color='r', linestyle='--', alpha=0.3)
                ax.text(0.02, 0.98, f'Best Val: {best_val:.4f} (epoch {best_val_epoch})',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
