import pandas as pd
import numpy as np
import re
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------------------
# 1. Load & clean data
# --------------------------------------

train_df = pd.read_csv(
    r"D:\coding\python\pyTorch\IIT project\Q4 SENTIMENT\sentiment analysis\train.csv",
    encoding='ISO-8859-1'
)

test_df = pd.read_csv(
    r"D:\coding\python\pyTorch\IIT project\Q4 SENTIMENT\sentiment analysis\test.csv",
    encoding='ISO-8859-1'
)

train_df = train_df[['text', 'sentiment']]
test_df = test_df[['text', 'sentiment']]  # Assuming test.csv also contains true labels

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train_df['text'] = train_df['text'].apply(clean_text)
test_df['text'] = test_df['text'].apply(clean_text)

# --------------------------------------
# 2. Label encoding
# --------------------------------------

label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['sentiment'])
test_df = test_df.dropna(subset=['sentiment'])
test_df['label'] = label_encoder.transform(test_df['sentiment'])

# --------------------------------------
# 3. Build vocabulary
# --------------------------------------

from collections import Counter

def build_vocab(texts, specials=["<PAD>", "<UNK>"]):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {token: idx for idx, token in enumerate(specials)}
    for word in counter:
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab

vocab = build_vocab(train_df["text"])
pad_idx = vocab["<PAD>"]
unk_idx = vocab["<UNK>"]

def text_to_tensor(text, vocab):
    return torch.tensor([vocab.get(token, unk_idx) for token in text.split()], dtype=torch.long)

# --------------------------------------
# 4. Dataset and DataLoader
# --------------------------------------

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = [text_to_tensor(text, vocab) for text in texts]
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_fn(batch):
    texts, labels = zip(*batch)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=pad_idx)
    return padded_texts, torch.tensor(labels, dtype=torch.long)


# --------------------------------------
# 5. Train/Val Split and Dataloaders
# --------------------------------------

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'], train_df['label'], test_size=0.2, random_state=42
)

train_dataset = SentimentDataset(train_texts, train_labels, vocab)
val_dataset = SentimentDataset(val_texts, val_labels, vocab)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

# --------------------------------------
# 6. Define LSTM Model
# --------------------------------------

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix=None, pad_idx=0):
        super(SentimentModel, self).__init__()

        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=pad_idx)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                            bidirectional=True, num_layers=2, dropout=0.3)

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)

        # Concatenate the final forward and backward hidden states
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.dropout(hidden_cat)
        return self.fc(out)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentModel(len(vocab), embed_dim=128, hidden_dim=128, output_dim=3).to(device)

# --------------------------------------
# 7. Train and Validate
# --------------------------------------

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_losses = []
val_accuracies = []
val_losses = []

def train_one_epoch():
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs, labels = [x.to(device) for x in batch]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    return avg_loss

def validate():
    model.eval()
    total_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = [x.to(device) for x in batch]
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(val_loader)
    val_losses.append(avg_loss)
    val_acc = correct / total
    val_accuracies.append(val_acc)
    return val_acc

for epoch in range(5):
    loss = train_one_epoch()
    acc = validate()
    print(f"Epoch {epoch+1} | Train Loss: {loss:.4f} | Val Acc: {acc:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='x')
plt.title("Train vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# --------------------------------------
# 8. Evaluate on Test Set
# --------------------------------------

test_texts = [text_to_tensor(text, vocab) for text in test_df['text']]
test_padded = pad_sequence(test_texts, batch_first=True, padding_value=pad_idx)

model.eval()
with torch.no_grad():
    outputs = model(test_padded.to(device))
    preds = torch.argmax(outputs, dim=1).cpu().numpy()

test_df['predicted'] = label_encoder.inverse_transform(preds)
true_labels = test_df['label'].values

# Accuracy & Report
acc = accuracy_score(true_labels, preds)
print(f"\n✅ Test Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(true_labels, preds, target_names=label_encoder.classes_))

# Confusion Matrix

plt.figure(figsize=(8, 5))
cm = confusion_matrix(true_labels, preds)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Test Set Confusion Matrix")
plt.show()

def predict_sentiment(text, model, vocab, label_encoder, device):
    model.eval()
    text = clean_text(text)
    tokens = text.split()
    tensor = torch.tensor([vocab.get(token, vocab["<UNK>"]) for token in tokens], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()
        sentiment = label_encoder.inverse_transform([pred])[0]
    return sentiment

# Example usage:
while True:
    user_input = input("\nEnter text (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    sentiment = predict_sentiment(user_input, model, vocab, label_encoder, device)
    print(f"Predicted Sentiment: {sentiment}")

