import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. SETUP DATA (Assuming X_train, y_train exist from Phase 1)
# Convert numpy arrays to PyTorch Tensors
# We use .float() because PyTorch defaults to float32
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1) # Shape becomes (N, 1)

X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(1)

# Create DataLoaders (Batching)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False) # Don't shuffle time series? 
# Actually: We CAN shuffle TRAINING batches, but NOT within the window. 
# Usually in finance, we shuffle batches to break correlation bias during training.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 2. SETUP MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMPredictor(input_dim=X_train.shape[2], hidden_dim=64, num_layers=2).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. TRAINING LOOP
epochs = 20
print(f"🚀 Starting Training on {device}...")

for epoch in range(epochs):
    model.train() # Set mode to training (enables Dropout)
    running_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # A. Zero Gradients
        optimizer.zero_grad()
        
        # B. Forward Pass
        predictions = model(batch_X)
        
        # C. Calculate Loss
        loss = criterion(predictions, batch_y)
        
        # D. Backward Pass (Calculate Gradients)
        loss.backward()
        
        # E. Update Weights
        optimizer.step()
        
        running_loss += loss.item()
    
    # Print average loss per epoch
    epoch_loss = running_loss / len(train_loader)
    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}")

print("✔ Training Complete.")

# 4. EVALUATION
model.eval() # Set mode to evaluation
with torch.no_grad():
    test_predictions = model(X_test_tensor.to(device)).cpu().numpy()
    
# Quick check
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot(y_test[:100], label='Actual Return')
plt.plot(test_predictions[:100], label='Predicted Return')
plt.title("LSTM Forecast (First 100 Test Days)")
plt.legend()
plt.show()