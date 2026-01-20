import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class TimeSeriesProcessor:
    def __init__(self, window_size=60, split_ratio=0.8):
        self.window_size = window_size
        self.split_ratio = split_ratio
        self.scaler = StandardScaler() # Standardize (Mean=0, Std=1)
        
    def train_test_split(self, df):
        """
        Splits data chronologically.
        NO SHUFFLING allowed in time series!
        """
        train_size = int(len(df) * self.split_ratio)
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]
        return train_data, test_data
        
    def scale_data(self, train_df, test_df):
        """
        Fits scaler on TRAIN, transforms both TRAIN and TEST.
        Prevents look-ahead bias.
        """
        # Fit on training data only
        self.scaler.fit(train_df)
        
        # Transform both
        train_scaled = pd.DataFrame(self.scaler.transform(train_df), 
                                    columns=train_df.columns, 
                                    index=train_df.index)
        
        test_scaled = pd.DataFrame(self.scaler.transform(test_df), 
                                   columns=test_df.columns, 
                                   index=test_df.index)
        
        return train_scaled, test_scaled
    
    def create_sequences(self, df, target_col='log_return'):
        """
        Converts DataFrame into 3D Arrays for the AI.
        Input Shape: (N_days, N_features)
        Output X Shape: (N_samples, Window_Size, N_features)
        Output y Shape: (N_samples,)
        """
        data_array = df.values
        # Find the index of the target column
        target_idx = df.columns.get_loc(target_col)
        
        X, y = [], []
        
        # Iterate through data creating sliding windows
        # We stop 'window_size' steps before the end
        for i in range(len(data_array) - self.window_size):
            # The Sequence (History)
            # From i to i+60
            seq_x = data_array[i : i + self.window_size]
            
            # The Target (Prediction)
            # The value at i+60 (The very next day)
            seq_y = data_array[i + self.window_size, target_idx]
            
            X.append(seq_x)
            y.append(seq_y)
            
        return np.array(X), np.array(y)

# --- EXECUTION & VERIFICATION ---
def run_pipeline_check(df_features):
    print("\n[Data Processor] Starting Tensor Transformation...")
    
    processor = TimeSeriesProcessor(window_size=60, split_ratio=0.8)
    
    # 1. Split
    train_df, test_df = processor.train_test_split(df_features)
    print(f"✔ Split: Train Rows={len(train_df)}, Test Rows={len(test_df)}")
    
    # 2. Scale
    train_scaled, test_scaled = processor.scale_data(train_df, test_df)
    print(f"✔ Scaled: Mean should be ~0. Train Mean: {train_scaled.mean().mean():.4f}")
    
    # 3. Create Sequences (The 3D Tensors)
    X_train, y_train = processor.create_sequences(train_scaled)
    X_test, y_test = processor.create_sequences(test_scaled)
    
    print("\n--- FINAL TENSOR SHAPES ---")
    print(f"X_train: {X_train.shape}  (Samples, TimeSteps, Features)")
    print(f"y_train: {y_train.shape}  (Samples,)")
    print(f"X_test:  {X_test.shape}")
    
    return X_train, y_train, X_test, y_test

# If running this standalone, you'd need the df_features from Lesson 3.
# This code assumes 'df_features' exists.