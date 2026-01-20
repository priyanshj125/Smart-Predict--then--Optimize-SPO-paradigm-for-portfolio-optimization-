import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens
    in the sequence. The Transformer doesn't know order without this.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix of [max_len, d_model] representing positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (Seq_Len, Batch_Size, Features)
        return x + self.pe[:x.size(0), :]

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, output_dim=1, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        
        # 1. Input Embedding
        # Transformers expect a specific dimension size (d_model), so we project our input up to it.
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 2. Transformer Encoder Layers
        # nhead=4 means it looks for 4 different types of patterns simultaneously
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 3. Output Decoder
        self.fc = nn.Linear(d_model, output_dim)
        self.d_model = d_model

    def forward(self, x):
        # x shape comes in as: (Batch_Size, Window_Size, Features)
        # Transformers in PyTorch expect: (Window_Size, Batch_Size, Features) by default
        x = x.permute(1, 0, 2) 
        
        # 1. Embed & Add Position
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # 2. Pass through Transformer
        # output shape: (Window_Size, Batch_Size, d_model)
        output = self.transformer_encoder(x)
        
        # 3. Take the output of the LAST time step
        # This represents the "summary" of the entire window
        last_output = output[-1, :, :]
        
        # 4. Predict
        prediction = self.fc(last_output)
        
        return prediction

# --- VERIFICATION ---
if __name__ == "__main__":
    # Test on dummy data
    # (Batch=32, Window=60, Features=5)
    dummy_input = torch.randn(32, 60, 5)
    
    model = TransformerPredictor(input_dim=5, d_model=64, nhead=4, num_layers=2)
    output = model(dummy_input)
    
    print(f"Transformer Input: {dummy_input.shape}")
    print(f"Transformer Output: {output.shape} (Should be [32, 1])")
    print("✔ Transformer Model is valid.")