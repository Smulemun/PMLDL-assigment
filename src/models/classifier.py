import torch.nn as nn

class ToxicWordsClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embedding_dim, embedding_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(embedding_dim // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(self.embedding(x))
        x = self.dropout(self.fc1(x))
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x