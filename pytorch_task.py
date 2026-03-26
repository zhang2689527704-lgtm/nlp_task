import torch
import os
from torch.utils.data import Dataset, DataLoader

abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
abc = abc + abc.lower() + "_"
device = "cuda" if torch.cuda.is_available() else "cpu"

def letter_index(lett):
    if lett in abc:
        return abc.find(lett)
    return abc.find("_")

def word_to_tensor(word):
    r = torch.zeros(len(word), len(abc))
    for index, lett in enumerate(word):
        r[index][letter_index(lett)] = 1
    return r

DATA_FOLDER = "names"

COUNTRY_LIST = [
    "Arabic", "Chinese", "Czech", "Dutch", "English",
    "French", "German", "Greek", "Irish", "Italian",
    "Japanese", "Korean", "Polish", "Portuguese", "Russian",
    "Scottish", "Spanish", "Vietnamese"
]

def load_data(data_folder):
    all_names = []
    all_label_indices = []
    
    print("data index：")
    for country_idx, country_name in enumerate(COUNTRY_LIST):
        file_path = os.path.join(data_folder, f"{country_name}.txt")
        
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            names = [line.strip() for line in f if line.strip()]
            
        all_names.extend(names)
        all_label_indices.extend([country_idx] * len(names))
        print(f"  [{country_idx}] {country_name}: {len(names)} names")
    return all_names, all_label_indices

class NameCountryDataset(Dataset):
    def __init__(self, data_folder):
        self.names, self.label_indices = load_data(data_folder)
        self.num_countries = len(COUNTRY_LIST)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        label_idx = self.label_indices[idx]

        name_tensor = word_to_tensor(name)

        country_tensor = torch.zeros(self.num_countries)
        country_tensor[label_idx] = 1

        return name_tensor, country_tensor

def collate_fn(batch):
    name_tensors, country_tensors = zip(*batch)
    max_len = max([t.shape[0] for t in name_tensors])
    vocab_size = name_tensors[0].shape[1]
    
    padded_names = torch.zeros(len(batch), max_len, vocab_size)
    for i, tensor in enumerate(name_tensors):
        padded_names[i, :tensor.shape[0], :] = tensor
        
    return padded_names, torch.stack(country_tensors)


if __name__ == "__main__":
    dataset = NameCountryDataset(DATA_FOLDER)
  
    test_name = "Khoury"
    if test_name in dataset.names:
        idx = dataset.names.index(test_name)
        name_tensor, country_tensor = dataset[idx]
        
        print(f"Test name: {test_name}")
        print(f"Name Tensor Shape: {name_tensor.shape}")
        print(f"Country vector: {country_tensor}")
        print(f"Country Index: {country_tensor.argmax().item()}")
        print(f"Country: {COUNTRY_LIST[country_tensor.argmax().item()]}")
        print(f" One-Hot :")
        print(name_tensor)


# Country: Arab
