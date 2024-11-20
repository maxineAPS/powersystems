from args import Args
from data_processing import load_and_preprocess_data, split_data
from training import train_ensemble
from evaluation import evaluate_ensemble, plot_predictions

# Load configurations
args = Args()

# Data processing
data, scaler = load_and_preprocess_data(args.file_path, args.target_columns, args.sample_rate)
X_train, y_train, X_test, y_test = split_data(data, args.sequence_length, args.split_ratio)

# Train ensemble
models = train_ensemble(X_train, y_train, args)

# Evaluate and plot
mean_predictions, std_predictions = evaluate_ensemble(models, X_test, scaler, args)
plot_predictions(mean_predictions, std_predictions, y_test, scaler, args)
