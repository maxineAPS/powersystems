import numpy as np
import matplotlib.pyplot as plt

def evaluate_ensemble(models, X_test, scaler, args):
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
            predictions.append(preds)

    predictions = np.array(predictions)  # Shape: (ensemble_size, num_samples, output_size)
    mean_predictions = predictions.mean(axis=0)
    std_predictions = predictions.std(axis=0)

    # Rescale predictions and ground truth
    mean_predictions_rescaled = scaler.inverse_transform(mean_predictions)
    std_predictions_rescaled = scaler.inverse_transform(std_predictions)

    return mean_predictions_rescaled, std_predictions_rescaled

def plot_predictions(mean_predictions, std_predictions, y_test, scaler, args):
    y_test_rescaled = scaler.inverse_transform(y_test)

    # Plotting
    time_points = np.arange(len(mean_predictions))[:args.num_to_plot] * args.time_window
    fig, axes = plt.subplots(1, len(args.target_columns), figsize=(18, 6), sharey=False)

    for idx, (ax, column) in enumerate(zip(axes, args.target_columns)):
        ax.plot(
            time_points,
            y_test_rescaled[:args.num_to_plot, idx],
            label='Ground Truth',
            color='blue'
        )
        ax.plot(
            time_points,
            mean_predictions[:args.num_to_plot, idx],
            label='Prediction',
            color='orange'
        )
        ax.fill_between(
            time_points,
            mean_predictions[:args.num_to_plot, idx] - std_predictions[:args.num_to_plot, idx],
            mean_predictions[:args.num_to_plot, idx] + std_predictions[:args.num_to_plot, idx],
            color='orange',
            alpha=0.2,
            label='Uncertainty (± std)'
        )
        ax.set_title(f'Predictions: {column}')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel(column)
        ax.grid()
        ax.legend()

    plt.tight_layout()
    plt.show()
