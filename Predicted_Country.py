import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import glob
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
abc = abc + abc.lower() + "_"
N_LETTERS = len(abc)
MAX_NAME_LENGTH = 20 

def letter_index(lett):
    if lett in abc:
        return abc.find(lett)
    return abc.find("_")

def word_to_tensor(word):
    tensor = torch.zeros(MAX_NAME_LENGTH, N_LETTERS)
    for li, letter in enumerate(word[:MAX_NAME_LENGTH]):
        tensor[li][letter_index(letter)] = 1
    return tensor

class NamesDataset(Dataset):
    def __init__(self, data_dir="names"):
        self.countries = [
            'Arabic', 'Chinese', 'Czech', 'Dutch', 'English',
            'French', 'German', 'Greek', 'Irish', 'Italian',
            'Japanese', 'Korean', 'Polish', 'Portuguese',
            'Russian', 'Scottish', 'Spanish', 'Vietnamese'
        ]
        self.country_to_idx = {c: i for i, c in enumerate(self.countries)}
        self.samples = self._load_data(data_dir)
        self.vocab_size = N_LETTERS
        self.max_len = MAX_NAME_LENGTH

    def _load_data(self, data_dir):
        samples = []
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            current_dir = os.getcwd()
        full_path = os.path.join(current_dir, data_dir)

        for file_path in glob.glob(os.path.join(full_path, "*.txt")):
            country = os.path.splitext(os.path.basename(file_path))[0]
            if country not in self.country_to_idx:
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    name = line.strip()
                    cleaned = [c for c in name if c.isalpha()]
                    name = ''.join(cleaned).strip()
                    if name:
                        samples.append((name, self.country_to_idx[country]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, country_idx = self.samples[idx]
        name_tensor = word_to_tensor(name)
        return name_tensor, torch.tensor(country_idx, dtype=torch.long)

class NameClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        _, hidden = self.rnn(x)
        out = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.fc(out)
        return out

def train_model(model, train_loader, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    print(f"Device: {device} | Starting training (total {epochs} epochs)...\n")
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            y_pred = model(x)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.argmax(y_pred, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            total_loss += loss.item()

        avg_loss = total_loss / total
        accuracy = correct / total
        print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

def predict_name(model, name, dataset):
    model.eval()
    with torch.no_grad():
        x = word_to_tensor(name).unsqueeze(0).to(device)
        output = model(x)
        pred_idx = torch.argmax(output, dim=1).item()
    return dataset.countries[pred_idx]

if __name__ == '__main__':
    dataset = NamesDataset(data_dir="names")
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    print(f"Dataset loading complete: {len(dataset)} names | 18 countries")

    model = NameClassifier(
        input_size=N_LETTERS,
        hidden_size=256,
        num_classes=len(dataset.countries)
    ).to(device)

    train_model(model, train_loader, epochs=20)

    print("\n Predictive Test ")
    test_names = ["Cui", "Smith", "Mohammed", "Ivanov", "Nguyen", "Pozhidaev"]
    for name in test_names:
        result = predict_name(model, name, dataset)
        print(f"name: {name:10} → Predicted Country: {result}")

      
# Epoch  1 | Loss: 0.0123 | Accuracy: 0.5521
# Epoch  2 | Loss: 0.0089 | Accuracy: 0.6649
# Epoch  3 | Loss: 0.0080 | Accuracy: 0.6943
# Epoch  4 | Loss: 0.0073 | Accuracy: 0.7183
# Epoch  5 | Loss: 0.0068 | Accuracy: 0.7374
# Epoch  6 | Loss: 0.0064 | Accuracy: 0.7537
# Epoch  7 | Loss: 0.0061 | Accuracy: 0.7637
# Epoch  8 | Loss: 0.0057 | Accuracy: 0.7779
# Epoch  9 | Loss: 0.0053 | Accuracy: 0.7934
# Epoch 10 | Loss: 0.0051 | Accuracy: 0.7958
# Epoch 11 | Loss: 0.0049 | Accuracy: 0.8080
# Epoch 12 | Loss: 0.0046 | Accuracy: 0.8157
# Epoch 13 | Loss: 0.0044 | Accuracy: 0.8228
# Epoch 14 | Loss: 0.0042 | Accuracy: 0.8320
# Epoch 15 | Loss: 0.0041 | Accuracy: 0.8365
# Epoch 16 | Loss: 0.0039 | Accuracy: 0.8411
# Epoch 17 | Loss: 0.0037 | Accuracy: 0.8505
# Epoch 18 | Loss: 0.0036 | Accuracy: 0.8534
# Epoch 19 | Loss: 0.0034 | Accuracy: 0.8601
# Epoch 20 | Loss: 0.0033 | Accuracy: 0.8640

#  Predictive Test 
# name: Cui        → Predicted Country: Chinese
# name: Smith      → Predicted Country: English
# name: Mohammed   → Predicted Country: English
# name: Ivanov     → Predicted Country: Russian
# name: Nguyen     → Predicted Country: Vietnamese
# name: Pozhidaev  → Predicted Country: Russian
