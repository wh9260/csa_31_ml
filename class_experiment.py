"""
Experiment class for running and tracking Keras model training experiments.
"""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Any
import keras
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


class experiment:
    """
    A class to manage Keras model training experiments.

    This class handles model training, evaluation, and result persistence.
    All training parameters are configurable and results are saved to the
    results directory with detailed metrics and training history.
    """

    def __init__(
        self,
        model: keras.Model,
        experiment_name: str,
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 1,
        **kwargs
    ):
        """
        Initialize an experiment.

        Args:
            model: Compiled Keras model to train
            experiment_name: Name for this experiment (used in result files)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of training data to use for validation
            verbose: Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)
            **kwargs: Additional parameters to pass to model.fit()
        """
        self.model = model
        self.experiment_name = experiment_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.verbose = verbose
        self.fit_kwargs = kwargs
        self.history = None
        self.test_results = None

        # Ensure results directory exists
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)

    def run(
        self,
        x_train: Any,
        y_train: Any,
        x_test: Any,
        y_test: Any
    ) -> Tuple[keras.callbacks.History, dict]:
        """
        Run the experiment: train and evaluate the model.

        Args:
            x_train: Training features
            y_train: Training labels
            x_test: Test features
            y_test: Test labels

        Returns:
            Tuple of (training history, test metrics dictionary)
        """
        print(f"\n{'='*60}")
        print(f"Running Experiment: {self.experiment_name}")
        print(f"{'='*60}\n")

        # Train the model
        print("Training model...")
        self.history = self.model.fit(
            x_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=self.verbose,
            **self.fit_kwargs
        )

        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_metrics = self.model.evaluate(x_test, y_test, verbose=self.verbose, return_dict=True)
        self.test_results = test_metrics

        # Save results
        self._save_results()

        print(f"\n{'='*60}")
        print(f"Experiment Complete: {self.experiment_name}")
        print(f"{'='*60}\n")

        return self.history, self.test_results

    def _save_results(self) -> str:
        """
        Save experiment results to disk.

        Returns:
            Path to the saved results file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"{self.experiment_name}_{timestamp}.json"
        results_path = self.results_dir / results_filename

        # Prepare results dictionary
        results = {
            "experiment_name": self.experiment_name,
            "timestamp": timestamp,
            "datetime": datetime.now().isoformat(),
            "parameters": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "validation_split": self.validation_split,
                **self.fit_kwargs
            },
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

        # Also save a human-readable summary
        summary_path = self.results_dir / f"{self.experiment_name}_{timestamp}_summary.txt"
        self._save_summary(summary_path, results)

        # Save training history plot
        plot_path = self.results_dir / f"{self.experiment_name}_{timestamp}_history.png"
        self._plot_history(plot_path)

        print(f"\nResults saved to:")
        print(f"  JSON: {results_path}")
        print(f"  Summary: {summary_path}")
        print(f"  Plot: {plot_path}")

        return str(results_path)

    def _get_model_summary(self) -> dict:
        """Get a dictionary representation of the model architecture."""
        total_params = sum([keras.ops.size(w).numpy() for w in self.model.trainable_weights])

        return {
            "total_parameters": int(total_params),
            "layers": len(self.model.layers),
            "input_shape": str(self.model.input_shape),
            "output_shape": str(self.model.output_shape)
        }

    def _save_summary(self, path: Path, results: dict) -> None:
        """Save a human-readable text summary of the experiment."""
        with open(path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write(f"EXPERIMENT SUMMARY: {self.experiment_name}\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Date/Time: {results['datetime']}\n\n")

            f.write("PARAMETERS:\n")
            f.write("-" * 70 + "\n")
            for key, value in results['parameters'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            f.write("MODEL ARCHITECTURE:\n")
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
            # Create a table of metrics per epoch
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
        """
        Plot training history and save as PNG.

        Creates subplots for each metric (loss, accuracy, etc.) showing
        both training and validation curves over epochs.
        """
        history = self.history.history

        # Separate metrics into training and validation
        train_metrics = {}
        val_metrics = {}

        for key, values in history.items():
            if key.startswith('val_'):
                val_metrics[key[4:]] = values  # Remove 'val_' prefix
            else:
                train_metrics[key] = values

        # Determine number of subplots needed
        metric_names = list(train_metrics.keys())
        num_metrics = len(metric_names)

        if num_metrics == 0:
            print("  Warning: No metrics to plot")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics))
        if num_metrics == 1:
            axes = [axes]  # Make it iterable

        epochs = range(1, len(train_metrics[metric_names[0]]) + 1)

        for idx, metric_name in enumerate(metric_names):
            ax = axes[idx]

            # Plot training metric
            ax.plot(epochs, train_metrics[metric_name], 'b-', label=f'Training {metric_name}', linewidth=2)

            # Plot validation metric if available
            if metric_name in val_metrics:
                ax.plot(epochs, val_metrics[metric_name], 'r-', label=f'Validation {metric_name}', linewidth=2)

            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric_name.capitalize(), fontsize=12)
            ax.set_title(f'{metric_name.capitalize()} over Epochs', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)

            # Add best value annotations
            best_train = max(train_metrics[metric_name]) if 'acc' in metric_name else min(train_metrics[metric_name])
            best_train_epoch = train_metrics[metric_name].index(best_train) + 1

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

        print(f"  Training history plot saved")
