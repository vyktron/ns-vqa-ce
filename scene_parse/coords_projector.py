import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import json

class ObjectLocalizationMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(ObjectLocalizationMLP, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    
def get_data(path : str = "scene_parse/CLEVR_mini/CLEVR_mini_coco_anns.json") -> tuple[list[list[int, int, int]], list[list[float, float]]] :

    train_input = [] ; val_input = []
    train_target = [] ; val_target = []

    # Open the file
    with open(path, "r") as f:
        annotations = json.load(f)
        scenes = annotations["scenes"]   
        for s in scenes :
            for o in s["objects"] :
                # One hot encoding of the shape (cube, sphere, or cylinder)
                shape = [1, 0, 0] if o["shape"] == "cube" else [0, 1, 0] if o["shape"] == "sphere" else [0, 0, 1]
                size = 0 if o["size"] == "small" else 1
                if s["split"] == "train" :
                    train_input.append(shape + [size, o["pixel_coords"][0], o["pixel_coords"][1]])
                    train_target.append([o["3d_coords"][0], o["3d_coords"][1]])
                else :
                    val_input.append(shape + [size, o["pixel_coords"][0], o["pixel_coords"][1]])
                    val_target.append([o["3d_coords"][0], o["3d_coords"][1]])
    
    return train_input, train_target, val_input, val_target
    
if __name__ == "__main__" :

    # Define the input size, hidden size, and output size
    input_size = 6  # 3 for shape, 1 for size (Large or Small) and 2 for pixel coordinates (x, y)
    hidden_size = [8,16,8]
    output_size = 2  # 2 for 3D coordinates (x, y)

    # Create the model
    model = ObjectLocalizationMLP(input_size, hidden_size, output_size)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

    # Get the data
    train_input, train_target, val_input, val_target = get_data()

    # Convert data to PyTorch tensors
    train_input = torch.tensor(train_input, dtype=torch.float32)
    train_target = torch.tensor(train_target, dtype=torch.float32)
    val_input = torch.tensor(val_input, dtype=torch.float32)
    val_target = torch.tensor(val_target, dtype=torch.float32)

    # Combine input and target data into TensorDataset
    train_dataset = TensorDataset(train_input, train_target)
    val_dataset = TensorDataset(val_input, val_target)

    # Create DataLoader for training and validation sets
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = ObjectLocalizationMLP(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # Training loop
    num_epochs = 100

    for epoch in range(num_epochs):
        # Training
        model.train()
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for val_inputs, val_targets in val_loader:
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_targets).item()

        # Average validation loss
        val_loss /= len(val_loader)

        scheduler.step()

        # Print progress
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

    # Save the model
    torch.save(model.state_dict(), "scene_parse/attr_model.pth")

    # Load the model
    model = ObjectLocalizationMLP(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load("scene_parse/attr_model.pth"))