import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pyepo
import cvxpy as cp
from hmmlearn.hmm import GaussianHMM
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. HMM Regime Module
# ==========================================
class HMMRegimeDetector:
    def __init__(self, n_components=2):
        self.n_components = n_components
        
    def fit_predict_rolling(self, prices_df, window=252):
        """
        Train HMM on a rolling window of historical log returns & volatility.
        Ensures NO LOOK-AHEAD BIAS by predicting step 't' only using data up to 't-1'.
        """
        returns = prices_df.pct_change().fillna(0)
        log_returns = np.log(1 + returns)
        volatility = log_returns.rolling(window=21).std().fillna(0)
        
        n_days = len(prices_df)
        regime_probs = np.zeros((n_days, self.n_components))
        
        # We take the mean across all assets to represent the 'market' state
        mean_log_returns = log_returns.mean(axis=1).values
        mean_volatility = volatility.mean(axis=1).values
        
        # HMM Observation Features: [log_return, vol]
        features = np.column_stack([mean_log_returns, mean_volatility])
        
        for t in range(window, n_days):
            # Train only on past data [t-window : t-1]
            train_features = features[t-window:t]
            hmm = GaussianHMM(n_components=self.n_components, covariance_type="diag", n_iter=100, random_state=42)
            try:
                # Suppress output to avoid spam during rolling window
                hmm.fit(train_features)
                
                # Predict regime posterior probabilities for the current state (based on trained HMM)
                # Since we want features available AT time t-1 to predict the regime at t
                posteriors = hmm.predict_proba(train_features)
                # The posterior corresponding to the latest observation represents the probability state
                regime_probs[t] = posteriors[-1] 
            except Exception:
                # Fallback to a uniform probability distribution if HMM fails to converge
                if t > 0:
                    regime_probs[t] = regime_probs[t-1]
                else:
                    regime_probs[t] = np.ones(self.n_components) / self.n_components
                    
        return regime_probs

# ==========================================
# 2. Optimization Layer (The 'Oracle')
# ==========================================
class PortfolioOptModel(pyepo.model.opt.optModel):
    """
    Mean-Variance Optimization defined as a PyEPO model.
    Includes Budget (sum=1), Long-only (w>=0) constraints, and penalties for
    Transaction Costs (gamma) and L2 Regularization (lambda_l2).
    """
    def __init__(self, num_assets, gamma=0.001, lambda_l2=0.01, lambda_cov=1.0):
        self.num_assets = num_assets
        super().__init__()
        self.gamma = gamma          # Transaction cost parameter
        self.lambda_l2 = lambda_l2  # L2 regularization parameter
        self.lambda_cov = lambda_cov# Risk penalty (variance)
        
        # Parameters updated dynamically during training/backtesting
        self.Sigma = np.eye(num_assets)
        self.w_prev = np.ones(num_assets) / num_assets
        
    def _getModel(self):
        # We are using cvxpy instead of Gurobi, so we construct the problem in solve()
        # PyEPO requires returning (model, variables). The length of variables determines num_cost.
        return None, [None] * self.num_assets
        
    def setObj(self, cost):
        # PyEPO standard: the problem MINIMIZES c^T x.
        # So our cost vector represents negative expected returns (-r).
        self.cost = cost
        
    def solve(self):
        """
        Solves the Markowitz problem. Called by PyEPO loss functions during forward/backward pass.
        """
        w = cp.Variable(self.num_assets)
        c = self.cost 
        
        # Portfolio Variance formulation
        variance = cp.quad_form(w, self.Sigma)
        
        # Transaction Costs: Gamma * ||w - w_prev||_1
        transaction_costs = cp.norm1(w - self.w_prev)
        
        # L2 Regularization
        l2_reg = cp.sum_squares(w)
        
        # Minimize Cost = c^T w (which is -r^T w) + Penalties
        objective = cp.Minimize(c @ w + self.lambda_cov * variance + self.gamma * transaction_costs + self.lambda_l2 * l2_reg)
        
        # Constraints: Portfolio budget sum to 1, non-negativity (long-only)
        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]
        
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS, verbose=False) # ECOS is robust for QP+L1
            sol = w.value
            if sol is None:
                return np.ones(self.num_assets) / self.num_assets, 0.0
            
            # Clean up numeric precision issues
            sol = np.clip(sol, 0, 1)
            sol = sol / sol.sum()
            return sol, prob.value
        except:
            return np.ones(self.num_assets) / self.num_assets, 0.0
            
    def copy(self):
        new_model = PortfolioOptModel(self.num_assets, self.gamma, self.lambda_l2, self.lambda_cov)
        new_model.Sigma = self.Sigma.copy()
        new_model.w_prev = self.w_prev.copy()
        return new_model
        
    def addConstr(self, coef, rhs):
        pass # Not dynamically adding constraints in this simple demo

# ==========================================
# 3. SPO Linear Predictor
# ==========================================
class SPORegimePredictor(nn.Module):
    def __init__(self, num_features, num_assets):
        super(SPORegimePredictor, self).__init__()
        # Linear map from [HMM Posteriors + Technical Indicators] -> Expected Returns
        self.linear = nn.Linear(num_features, num_assets)
        
    def forward(self, x):
        return self.linear(x)

# ==========================================
# 4. Feature Engineering
# ==========================================
def prepare_features(prices_df, regime_probs):
    """
    Concatenate HMM Regimes with standard Technical Indicators (SMA, RSI approximation).
    """
    returns = prices_df.pct_change().fillna(0)
    sma_20 = prices_df.rolling(window=20).mean() / prices_df - 1
    sma_20 = sma_20.fillna(0)
    
    num_assets = prices_df.shape[1]
    features = []
    
    for t in range(len(prices_df)):
        # Technical indicators flattened for all assets: [Returns(N), SMA(N)]
        tech_inds = np.concatenate([returns.iloc[t].values, sma_20.iloc[t].values])
        # Include HMM Regime Probabilities
        hmm_state = regime_probs[t]
        
        combined_features = np.concatenate([hmm_state, tech_inds])
        features.append(combined_features)
        
    return np.array(features)

# ==========================================
# 5. Training Loop
# ==========================================
def train_spo_predictor(predictor, optmodel, features, actual_returns, sigmas, epochs=10, lr=1e-3):
    """
    Trains predictor using SPO+ Loss.
    """
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    
    # We use SPOPlus from PyEPO. It solves the Oracle to compute gradients.
    # Note: processes=1 avoids multiprocessing conflicts with PyTorch/cvxpy on some platforms
    spo_plus_loss = pyepo.func.SPOPlus(optmodel, processes=1)
    
    predictor.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        # Iterate sequentially to update w_prev (to respect path-dependent transaction costs)
        for i in range(len(features)):
            optimizer.zero_grad()
            
            x = torch.tensor(features[i], dtype=torch.float32).unsqueeze(0)
            
            # True Cost is negative return
            true_c = -torch.tensor(actual_returns[i], dtype=torch.float32).unsqueeze(0)
            
            # Set dynamic state parameters for the optimization oracle
            optmodel.Sigma = sigmas[i]
            if i > 0:
                # Normally you'd keep track of w_prev from the network's implied prediction
                # In SPO loop, keeping it fixed to a uniform vector for simplicity during batch training
                optmodel.w_prev = np.ones(optmodel.num_assets) / optmodel.num_assets
                
            # Compute true optimal weights
            optmodel.setObj(true_c[0].numpy())
            true_w_np, true_obj_np = optmodel.solve()
            true_w = torch.tensor(true_w_np, dtype=torch.float32).unsqueeze(0)
            true_obj = torch.tensor([true_obj_np], dtype=torch.float32).unsqueeze(0)
            
            # Predict
            pred_returns = predictor(x)
            pred_c = -pred_returns
            
            # SPO+ Loss requires: (Predicted Cost, True Cost, True Optimal Weights, True Objective)
            try:
                loss = spo_plus_loss(pred_c, true_c, true_w, true_obj)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            except Exception as e:
                pass # ECOS/CVXPY might fail to solve some specific continuous relaxations under SPO+ perturbation
            
        print(f"Epoch {epoch+1}/{epochs} | SPO+ Loss: {epoch_loss/len(features):.4f}")
        
    return predictor

# ==========================================
# 6. Backtest
# ==========================================
def run_enhanced_backtest(prices_df, predictor, features, start_idx=252):
    """
    Rolling window backtest evaluating Cumulative Returns and Sharpe Ratio.
    """
    num_assets = prices_df.shape[1]
    optmodel = PortfolioOptModel(num_assets=num_assets, gamma=0.005, lambda_l2=0.01, lambda_cov=0.5)
    
    portfolio_weights = np.zeros((len(prices_df), num_assets))
    portfolio_returns = np.zeros(len(prices_df))
    w_prev = np.ones(num_assets) / num_assets
    
    predictor.eval()
    with torch.no_grad():
        for t in range(start_idx, len(prices_df)):
            # Update Covariance (Past 21 days real volatility)
            recent_returns = prices_df.iloc[t-21:t].pct_change().dropna()
            if len(recent_returns) > 5:
                optmodel.Sigma = recent_returns.cov().values
                
            optmodel.w_prev = w_prev
            
            x = torch.tensor(features[t], dtype=torch.float32).unsqueeze(0)
            pred_c = -predictor(x).squeeze(0).numpy()
            
            optmodel.setObj(pred_c)
            w, _ = optmodel.solve()
            portfolio_weights[t] = w
            
            actual_return = prices_df.iloc[t] / prices_df.iloc[t-1] - 1
            port_ret = np.dot(w, actual_return.fillna(0).values)
            portfolio_returns[t] = port_ret
            w_prev = w
            
    bt_returns = portfolio_returns[start_idx:]
    cum_returns = np.cumprod(1 + bt_returns) - 1
    sharpe_ratio = np.mean(bt_returns) / (np.std(bt_returns) + 1e-8) * np.sqrt(252)
    
    return bt_returns, cum_returns, sharpe_ratio

# ==========================================
# Example Execution Mock-up
# ==========================================
if __name__ == "__main__":
    # 1. Generate Dummy Market Data
    np.random.seed(42)
    days = 300
    assets = 5
    prices = pd.DataFrame(
        np.exp(np.cumsum(np.random.normal(0.0001, 0.015, (days, assets)), axis=0)),
        columns=[f"Asset_{i}" for i in range(assets)]
    )
    
    print("1. Extracting HMM Regimes...")
    hmm = HMMRegimeDetector(n_components=2)
    regime_probs = hmm.fit_predict_rolling(prices, window=50) # Small window for demo
    
    print("2. Preparing Features...")
    features = prepare_features(prices, regime_probs)
    actual_returns = prices.pct_change().fillna(0).values
    
    # Store dynamic sigmas
    sigmas = []
    for t in range(days):
        if t < 21:
            sigmas.append(np.eye(assets))
        else:
            sigmas.append(prices.iloc[t-21:t].pct_change().dropna().cov().values)
            
    print("3. Initializing SPO Model and Predictor...")
    optmodel = PortfolioOptModel(num_assets=assets, gamma=0.005, lambda_l2=0.01, lambda_cov=0.5)
    num_features = features.shape[1]
    predictor = SPORegimePredictor(num_features=num_features, num_assets=assets)
    
    print("4. Training SPO Predictor (Running small epoch for demo)...")
    train_start, train_end = 50, 200
    predictor = train_spo_predictor(
        predictor, 
        optmodel, 
        features[train_start:train_end], 
        actual_returns[train_start:train_end], 
        sigmas[train_start:train_end],
        epochs=2
    )
    
    print("5. Running Backtest...")
    bt_rets, cum_rets, sharpe = run_enhanced_backtest(prices, predictor, features, start_idx=200)
    
    print("-" * 40)
    print(f"Backtest Completed!")
    print(f"Final Cumulative Return: {cum_rets[-1]*100:.2f}%")
    print(f"Annualized Sharpe Ratio: {sharpe:.2f}")
    print("-" * 40)
