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
        
        # Input Embedding
        # Transformers expect a specific dimension size (d_model), so we project our input up to it.
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder Layers
        # nhead=4 means it looks for 4 different types of patterns simultaneously
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        #  Output Decoder
        self.fc = nn.Linear(d_model, output_dim)
        self.d_model = d_model

    def forward(self, x):
  
        x = x.permute(1, 0, 2) 
     
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
       
        output = self.transformer_encoder(x)
    
        last_output = output[-1, :, :]
        
      
        prediction = self.fc(last_output)
        
        return prediction


if __name__ == "__main__":
  
    dummy_input = torch.randn(32, 60, 5)
    
    model = TransformerPredictor(input_dim=5, d_model=64, nhead=4, num_layers=2)
    output = model(dummy_input)
    
    print(f"Transformer Input: {dummy_input.shape}")
    print(f"Transformer Output: {output.shape} (Should be [32, 1])")
    print("✔ Transformer Model is valid.")
