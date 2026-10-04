import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# 1. Konfiguracja
DATA_PATH = Path("data/dataset.txt")
SEQ_LENGTH = 30  # Długość kontekstu (ile znaków wstecz widzi sieć)
BATCH_SIZE = 128  # Wielkość paczki danych
EMBED_DIM = 64  # Wymiar reprezentacji znaku
HIDDEN_DIM = 128  # Liczba neuronów w warstwie ukrytej
EPOCHS = 10  # Liczba epok treningu
LR = 0.003  # Szybkość uczenia (Learning Rate)

# Wybór urządzenia (na t3.micro będzie to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--> Używane urządzenie: {device}")

# 2. Wczytanie i przygotowanie tekstu
if not DATA_PATH.exists():
    raise FileNotFoundError("Brak pliku data/dataset.txt! Najpierw pobierz dane.")

print("--> Wczytywanie tekstu...")
with open(DATA_PATH, "r", encoding="utf-8", errors="ignore") as f:
    # Wczytujemy fragment lub całość (dla t3.micro przycięcie do 500k znaków zapobiegnie spowolnieniu)
    text = f.read(500_000)

vocab = sorted(list(set(text)))
vocab_size = len(vocab)
print(f"--> Rozmiar alfabetu (unikalne znaki): {vocab_size}")

# Mapowanie: znak <-> liczba
char_to_ix = {ch: i for i, ch in enumerate(vocab)}
ix_to_char = {i: ch for i, ch in enumerate(vocab)}

# Zamiana całego tekstu na postać numeryczną
data = torch.tensor([char_to_ix[ch] for ch in text], dtype=torch.long)


# 3. Definicja architektury sieci neuronowej (MLP z Embeddingiem)
class CharMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, seq_len, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Wejście do warstwy ukrytej to spłaszczony wektor (seq_len * embed_dim)
        self.fc1 = nn.Linear(seq_len * embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        out = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        out = out.view(
            out.size(0), -1
        )  # spłaszczenie do (batch_size, seq_len * embed_dim)
        out = self.relu(self.fc1(out))  # (batch_size, hidden_dim)
        out = self.fc2(out)  # (batch_size, vocab_size)
        return out


model = CharMLP(vocab_size, EMBED_DIM, SEQ_LENGTH, HIDDEN_DIM).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# 4. Funkcja generatora danych (Batching)
def get_batches(data, seq_len, batch_size):
    num_samples = len(data) - seq_len
    indices = list(range(num_samples))
    random.shuffle(indices)

    for i in range(0, len(indices) - batch_size, batch_size):
        batch_indices = indices[i : i + batch_size]
        X = torch.stack([data[idx : idx + seq_len] for idx in batch_indices])
        Y = torch.stack([data[idx + seq_len] for idx in batch_indices])
        yield X.to(device), Y.to(device)


# 5. Pętla treningowa
print("--> Rozpoczynamy trening sieci neuronowej...")
model.train()

for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    batches_count = 0

    for X_batch, Y_batch in get_batches(data, SEQ_LENGTH, BATCH_SIZE):
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, Y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches_count += 1

    avg_loss = total_loss / max(1, batches_count)
    print(f"Epoka [{epoch}/{EPOCHS}] | Srednia strata (Loss): {avg_loss:.4f}")


# 6. Generowanie nowego tekstu przy użyciu wytrenowanej sieci
def generate_text(model, start_str, length_to_generate=150):
    model.eval()
    curr_str = start_str.lower()

    # Dopełnienie początkowego ciągu, jeśli jest za krótki
    while len(curr_str) < SEQ_LENGTH:
        curr_str = " " + curr_str

    result = start_str

    with torch.no_grad():
        for _ in range(length_to_generate):
            input_seq = curr_str[-SEQ_LENGTH:]
            input_tensor = torch.tensor(
                [[char_to_ix.get(ch, 0) for ch in input_seq]], dtype=torch.long
            ).to(device)

            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=-1)

            # Losowanie kolejnego znaku według rozkładu prawdopodobieństwa
            next_char_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = ix_to_char[next_char_idx]

            result += next_char
            curr_str += next_char

    return result


print("\n--- Wygenerowany tekst przez sieć neuronową ---")
sample_prompt = "the "
print(generate_text(model, start_str=sample_prompt, length_to_generate=200))
