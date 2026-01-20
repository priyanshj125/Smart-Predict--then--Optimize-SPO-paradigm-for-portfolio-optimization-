import torch
import cvxpy as cp
import yfinance as yf
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

print("--- SYSTEM DIAGNOSTIC START ---")

# 1. TEST DATA CONNECTION
print("\n[1/3] Testing Data Connection (yfinance)...")
try:
    ticker = "AAPL"
    data = yf.download(ticker, period="5d", progress=False)
    if not data.empty:
        print(f"✔ Success: Downloaded {len(data)} rows for {ticker}.")
        print(f"  Last Close: ${data['Close'].iloc[-1].item():.2f}")
    else:
        print("✘ Error: Data is empty.")
except Exception as e:
    print(f"✘ Error: {e}")

# 2. TEST DEEP LEARNING ENGINE (PyTorch)
print("\n[2/3] Testing AI Engine (PyTorch)...")
try:
    x = torch.tensor([2.0], requires_grad=True)
    y = x ** 3  # Function: y = x^3
    y.backward() # Derivative: 3x^2 -> 3(2)^2 = 12
    grad = x.grad.item()
    if grad == 12.0:
        print(f"✔ Success: Gradient calculated correctly. PyTorch {torch.__version__} is active.")
    else:
        print(f"✘ Error: Gradient mismatch. Expected 12.0, got {grad}")
except Exception as e:
    print(f"✘ Error: {e}")

# 3. TEST OPTIMIZATION ENGINE (Cvxpy)
print("\n[3/3] Testing Solver Engine (Cvxpy)...")
try:
    x = cp.Variable()
    obj = cp.Minimize((x - 4)**2) # Minimum should be at x=4
    prob = cp.Problem(obj)
    prob.solve()
    if np.isclose(x.value, 4.0, atol=1e-4):
        print(f"✔ Success: Solver found optimal x = {x.value:.4f}")
    else:
        print(f"✘ Error: Solver failed. Result: {x.value}")
except Exception as e:
    print(f"✘ Error: {e}")

print("\n--- SYSTEM DIAGNOSTIC COMPLETE ---")