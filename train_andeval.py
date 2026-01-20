import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy

# --- SETUP (Using the classes and data you already have) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert data to tensors
train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float().unsqueeze(1))
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

def train_and_eval(model_class, name, input_dim, epochs=10):
    print(f"\n🥊 Training Contender: {name}...")
    
    # Initialize
    model = model_class(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Quick Training Loop
    model.train()
    for ep in range(epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            
    # Evaluate
    rmse, acc, ic = evaluate_model(model, X_test, y_test, model_name=name)
    return model, ic

# 1. Train LSTM
# Note: Ensure LSTMPredictor class is defined from Lesson 1
lstm_model, lstm_ic = train_and_eval(
    lambda input_dim: LSTMPredictor(input_dim, 64, 2), 
    "LSTM", 
    X_train.shape[2]
)

# 2. Train Transformer
# Note: Ensure TransformerPredictor class is defined from Lesson 3
tft_model, tft_ic = train_and_eval(
    lambda input_dim: TransformerPredictor(input_dim, 64, 4, 2), 
    "Transformer", 
    X_train.shape[2]
)

# 3. Select Winner
print("\n --- THE RESULT ---")
if tft_ic > lstm_ic:
    print(f"Transformer Wins! (IC: {tft_ic:.4f} vs {lstm_ic:.4f})")
    best_model = tft_model
    torch.save(best_model.state_dict(), "best_model.pth")
else:
    print(f"LSTM Wins! (IC: {lstm_ic:.4f} vs {tft_ic:.4f})")
    best_model = lstm_model
    torch.save(best_model.state_dict(), "best_model.pth")

print("✔ Best model saved as 'best_model.pth'. We will use this for Phase 3.")