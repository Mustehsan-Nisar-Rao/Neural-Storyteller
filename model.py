import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
   
    def forward(self, features):
        out = self.fc(features)
        out = self.relu(out)
        h = out.unsqueeze(0)
        c = torch.zeros_like(h).to(features.device)
        return h, c

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.3, pad_id=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
   
    def forward(self, captions, hidden):
        embeddings = self.embed(captions)
        out, hidden = self.lstm(embeddings, hidden)
        out = self.dropout(out)
        out = self.fc(out)
        return out, hidden

class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
   
    def forward(self, features, captions):
        h, c = self.encoder(features)
        outputs, _ = self.decoder(captions, (h, c))
        return outputs
