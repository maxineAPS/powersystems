import torch
from torch.utils.data import DataLoader, Dataset

class WindDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_ensemble(X_train, y_train, args):
    train_dataset = WindDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    models = []
    for i in range(args.ensemble_size):
        print(f"Training model {i+1}/{args.ensemble_size}...")
        model = SimpleLSTM(args.input_size, args.hidden_size, args.num_layers, args.output_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.MSELoss()

        # Train the model
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            print(f"Model {i+1}, Epoch {epoch+1}/{args.epochs}, Loss: {train_loss:.4f}")

        models.append(model)

    return models
