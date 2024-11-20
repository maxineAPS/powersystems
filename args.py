# args.py
class Args:
    # Data configurations
    file_path = '/Users/maxine/Downloads/hawaii2020/198325_21.66_-156.16_2019.csv'
    target_columns = [
        'wind speed at 100m (m/s)',
        'wind direction at 100m (deg)',
        'turbulent kinetic energy at 100m (m2/s2)'
    ]
    time_resolution = 5  # Resolution of data in minutes
    time_window = 60  # Future prediction time in minutes
    sequence_length = 12  # Input sequence length (number of steps in input)
    sample_rate = time_window // time_resolution  # Steps to skip for subsampling
    split_ratio = 0.8  # Train-test split ratio

    # Model configurations
    hidden_size = 64
    num_layers = 2
    ensemble_size = 3
    input_size = len(target_columns)
    output_size = len(target_columns)

    # Training configurations
    batch_size = 64
    learning_rate = 0.01
    epochs = 10

    # Plotting configurations
    num_to_plot = 50
