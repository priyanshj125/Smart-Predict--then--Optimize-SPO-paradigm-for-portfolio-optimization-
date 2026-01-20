import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Computes institutional metrics for a given model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval() # Turn off dropout
    
    # 1. Generate Predictions
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_test).float().to(device)
        # Ensure y_test is 1D for metric calculation
        y_true = y_test.flatten()
        
        # Get predictions
        y_pred = model(X_tensor).cpu().numpy().flatten()
    
    # 2. Compute Metrics
    
    # A. RMSE (Error Magnitude)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # B. Directional Accuracy (Hit Rate)
    # We compare the signs: +1 vs +1, -1 vs -1
    true_signs = np.sign(y_true)
    pred_signs = np.sign(y_pred)
    accuracy = np.mean(true_signs == pred_signs)
    
    # C. Information Coefficient (IC)
    # Correlation between prediction and reality
    ic, p_value = pearsonr(y_pred, y_true)
    
    # 3. Report
    print(f"--- 📊 Performance Report: {model_name} ---")
    print(f"1. RMSE (Lower is better):       {rmse:.6f}")
    print(f"2. Directional Accuracy (>50%):  {accuracy:.2%}")
    print(f"3. IC (Correlation > 0.01):      {ic:.4f} (p={p_value:.4f})")
    
    # Visual check of the first 50 days
    return rmse, accuracy, ic

# --- EXECUTION BLOCK ---
# Assuming you have 'model' (Transformer) and 'model_lstm' (LSTM) trained in memory
# and X_test, y_test from Phase 1.


# let's assume 'model' is your trained Transformer.

# rmse, acc, ic = evaluate_model(model, X_test, y_test, model_name="Transformer")