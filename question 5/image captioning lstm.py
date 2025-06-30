import os
import nltk
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import Counter
import string
import torchvision.models as models
import pandas as pd

nltk.download('punkt')

# ----------------------------
# Vocabulary Class
# ----------------------------
class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.index = 4

    def __len__(self):
        return len(self.word2idx)

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            tokens = nltk.tokenize.word_tokenize(sentence.lower())
            frequencies.update(tokens)

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.word2idx[word] = self.index
                self.idx2word[self.index] = word
                self.index += 1

    def __getitem__(self, word):
        return self.word2idx.get(word, self.word2idx["<unk>"])

    def idx2word_func(self, idx):
        return self.idx2word.get(idx, "<unk>")

# ----------------------------
# Flickr8k Dataset
# ----------------------------
class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, captions_csv, vocab, transform=None, max_len=50):
        self.root_dir = root_dir
        self.vocab = vocab
        self.transform = transform
        self.max_len = max_len
        self.data = []

        df = pd.read_csv(captions_csv, delimiter='|')
        df.columns = [col.strip() for col in df.columns]

        # Drop rows with missing values and convert to string
        df = df.dropna(subset=['image_name', 'comment'])
        df['image_name'] = df['image_name'].astype(str)
        df['comment'] = df['comment'].astype(str)

        for _, row in df.iterrows():
            img = row['image_name'].strip()
            caption = row['comment'].strip().lower().translate(str.maketrans('', '', string.punctuation))
            self.data.append((img, caption))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, caption = self.data[index]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(caption)
        caption_idx = [self.vocab["<start>"]] + [self.vocab[token] for token in tokens] + [self.vocab["<end>"]]
        caption_tensor = torch.tensor(caption_idx, dtype=torch.long)
        return image, caption_tensor

# ----------------------------
# Encoder
# ----------------------------
def get_encoder():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    modules = list(model.children())[:-1]
    return nn.Sequential(*modules)

# ----------------------------
# Decoder (LSTM)
# ----------------------------
class CaptionLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(CaptionLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.feat_map = nn.Linear(2048, embed_size)  # NEW: to project ResNet output
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        features = self.feat_map(features)  # [B, embed_size]
        features = features.unsqueeze(1)    # [B, 1, embed_size]
        embeddings = self.embed(captions)   # [B, seq_len, embed_size]
        embeddings = torch.cat((features, embeddings), dim=1)  # [B, 1 + seq_len, embed_size]
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, features, max_len=20):
        sampled_ids = []
        states = None
        features = self.feat_map(features).unsqueeze(1)  # project + add time dimension
        inputs = features
        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted).unsqueeze(1)
        return sampled_ids


# ----------------------------
# Collate Function
# ----------------------------
def collate_fn(batch, pad_idx):
    images, captions = zip(*batch)
    images = torch.stack(images, 0)
    lengths = [len(c) for c in captions]
    captions_padded = nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=pad_idx)
    return images, captions_padded, lengths

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    embed_size = 256
    hidden_size = 512
    num_epochs = 3
    learning_rate = 1e-3
    batch_size = 32

    captions_file = 'Q5/Flickr dataset/flickr30k_images/results.csv'
    image_dir = 'Q5/Flickr dataset/flickr30k_images/flickr30k_images'

    df = pd.read_csv(captions_file, delimiter='|')
    df.columns = [col.strip() for col in df.columns]
    df = df.dropna(subset=['comment'])
    all_captions = df['comment'].astype(str).tolist()

    vocab = Vocabulary()
    vocab.build_vocab(all_captions)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    dataset = Flickr8kDataset(image_dir, captions_file, vocab, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=lambda batch: collate_fn(batch, pad_idx=0))

    encoder = get_encoder()
    decoder = CaptionLSTM(embed_size, hidden_size, len(vocab), num_layers=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, decoder = encoder.to(device), decoder.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)

    for epoch in range(num_epochs):
        decoder.train()
        encoder.train()
        for idx, (imgs, captions, lengths) in enumerate(dataloader):
            imgs, captions = imgs.to(device), captions.to(device)
            features = encoder(imgs).squeeze(-1).squeeze(-1)
            outputs = decoder(features, captions[:, :-1])

            lengths = [l - 1 for l in lengths]
            targets = nn.utils.rnn.pack_padded_sequence(captions[:, 1:], lengths, batch_first=True, enforce_sorted=False)[0]
            outputs = nn.utils.rnn.pack_padded_sequence(outputs, lengths, batch_first=True, enforce_sorted=False)[0]

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

    # Save model and vocab
    torch.save(decoder.state_dict(), 'caption_lstm.pth')
    torch.save(vocab.__dict__, 'vocab.pth')

    # ----------------------------
    # Caption Generation
    # ----------------------------
    def generate_caption(image_path):
        decoder.eval()
        encoder.eval()
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = encoder(image).squeeze(-1).squeeze(-1)
            output_ids = decoder.sample(features)
            caption = []
            for idx in output_ids:
                word = vocab.idx2word_func(idx)
                if word == "<end>":
                    break
                caption.append(word)
            return ' '.join(caption)

    test_image = 'D:/coding/python/pyTorch/IIT project/Q5/Flickr dataset/flickr30k_images/flickr30k_images/36979.jpg'
    print("Generated Caption:", generate_caption(test_image))
