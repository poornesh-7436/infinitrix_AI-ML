import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import string
import torch.nn.functional as F



print("✅ CRNN - FULL 330K DATASET VERSION - FIXED")
print("70% Train + 30% Val | 10K Test Eval | num_workers=0")



# =============================================================================
# DEVICE SETUP
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
 


# =============================================================================
# DATASET CLASS
# =============================================================================
class NameDataset(Dataset):
     
      #  Custom Dataset for the Written Name dataset.
       # Expects:
       #  - Excel file with columns: 'FILENAME', 'IDENTITY'
       #  - Images in a folder named 'train/' (for training) 
    
     
    def __init__(self, df=None, excel_path="written_name_train_v2.xlsx"):
        if df is None:
            df = pd.read_excel(excel_path)

            # Drop rows with missing filename or label
        self.data = df.dropna(subset=['FILENAME','IDENTITY'])  

         # Normalize labels: uppercase, limit to first 6 characters (helps convergence)
        self.data['IDENTITY'] = self.data['IDENTITY'].str.upper().str[:6]

        # Reset index for clean indexing
        self.data = self.data.reset_index(drop=True) 

        print(f"Dataset loaded: {len(self.data):,} samples (FULL)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

          # Some filenames have ':' which is invalid on Windows → replace
        fname = str(row['FILENAME']).replace(':','_').replace('.png','')
        
        # Load image, fallback to gray placeholder if missing/corrupted
        try:
            img = Image.open(f"train/{fname}.png").convert('L')
        except:
            img = Image.new('L', (100,32), 128)

             # Fixed height 32, width 100
               # → [1, 32, 100]
            # → range [-1, 1]
        
        transform = T.Compose([T.Resize((32,100)), T.ToTensor(), T.Normalize(0.5,0.5)])
        img = transform(img)
        



        chars = string.ascii_uppercase + ' '
        label = torch.tensor([chars.find(c)+1 if c in chars else 0 for c in row['IDENTITY']])
        return img, label


# =============================================================================
# COLLATE FUNCTION - pads variable-length labels in a batch
# =============================================================================
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    return images, labels



# FULL 330K SPLIT (70/30)

# =============================================================================
# LOAD FULL TRAINING DATA & SPLIT 70/30
# =============================================================================
print("🔄 Loading FULL training dataset...")
full_train_dataset = NameDataset()  # Loads ALL ~330K rows
train_size = int(0.7 * len(full_train_dataset))  # ~231K train
val_size = len(full_train_dataset) - train_size   # ~99K val
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

print(f"✅ Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

# DataLoaders - num_workers=0 to avoid multiprocessing issues on some systems
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, 
                         collate_fn=collate_fn, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, 
                       collate_fn=collate_fn, num_workers=0, pin_memory=True)



# =============================================================================
# MODEL DEFINITION - Simple CRNN (CNN + GRU + FC)
# =============================================================================
class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
         # Feature extractor CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2,1)),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU()
        )
         # After CNN: [B, 128, 8, 50] → reshape to sequence of length 50, feature 128*8=1024
        self.rnn = nn.GRU(128*8, 128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, 28)

    def forward(self, x):
        x = self.cnn(x)                      # [B, 128, 8, 50]
        b, c, h, w = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, w, -1)            # [B, 50, 1024]
        x, _ = self.rnn(x)         # [B, 50, 256]
        return self.fc(x)        # [B, 50, 28]


# =============================================================================
# TRAINING SETUP
# =============================================================================
model = CRNN().to(DEVICE)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



print("\n🚀 Training FULL 330K dataset...")
print("Epoch | Train Loss | Val Loss")
print("-----|------------|--------")


best_val_loss = float('inf')


for epoch in range(15):  # More epochs for massive data
    # TRAIN (~231K samples)
    model.train()
    train_loss = 0
    for images, targets in train_loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(images)
        
        input_lengths = torch.full((images.size(0),), logits.size(1), dtype=torch.long)
        target_lengths = torch.full((images.size(0),), targets.size(1), dtype=torch.long)
        log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    
   # --------------------- VALIDATION ---------------------
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            logits = model(images)
            input_lengths = torch.full((images.size(0),), logits.size(1), dtype=torch.long)
            target_lengths = torch.full((images.size(0),), targets.size(1), dtype=torch.long)
            log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            val_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    print(f"{epoch+1:2d}   | {avg_train_loss:.3f}     | {avg_val_loss:.3f}")
    

     # Save best model based on validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_crnn_330k.pth')


# Save final model
torch.save(model.state_dict(), 'crnn_final_330k.pth')
print("\n🎉 FULL Training complete!")


# =============================================================================
# TEST EVALUATION ON 10K SAMPLES
# =============================================================================
print("\n🔍 10K TEST EVALUATION...")
chars = string.ascii_uppercase + ' '


def ctc_greedy_decode(logits, chars):
     #Simple greedy CTC decoding (collapse repeats, remove blanks)
    probs = F.softmax(logits, dim=2)
    pred_indices = torch.argmax(probs, dim=2)
    decoded = []
    for pred in pred_indices:
        pred = pred.cpu().numpy()
        last_char = -1
        word = []
        for p in pred:
            if p != 0 and p != last_char:
                word.append(p)
                last_char = p
        decoded.append(''.join([chars[i-1] for i in word if i > 0]))
    return decoded


def decode_target(target, chars):
    target = target[target != 0].cpu().numpy()
    return ''.join([chars[i-1] for i in target if i > 0])


def compute_cer(preds, targets):
    total, correct = 0, 0
    for p, t in zip(preds, targets):
        total += len(t)
        correct += sum(1 for a,b in zip(p, t) if a == b)
    return correct / total if total > 0 else 0


def compute_wer(preds, targets):
    correct = sum(1 for p,t in zip(preds, targets) if p.strip() == t.strip())
    return correct / len(preds) if len(preds) > 0 else 0


# Load test set (only first 10K samples)
test_df = pd.read_excel("written_name_test_v2.xlsx")
test_df_limited = test_df.head(10000)  # Exactly 10K test samples
test_dataset = NameDataset(test_df_limited)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, 
                        collate_fn=collate_fn, num_workers=0, pin_memory=True)

# Evaluation
model.eval()
test_loss = 0
all_preds, all_targets = [], []


print("Evaluating 10K test samples...")
with torch.no_grad():
    for batch_idx, (images, targets) in enumerate(test_loader):
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        
        logits = model(images)
        
        input_lengths = torch.full((images.size(0),), logits.size(1), dtype=torch.long)
        target_lengths = torch.full((images.size(0),), targets.size(1), dtype=torch.long)
        log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        test_loss += loss.item()
        
        preds = ctc_greedy_decode(logits, chars)
        all_preds.extend(preds)
        all_targets.extend([decode_target(t, chars) for t in targets])
        
        if batch_idx % 20 == 0:
            print(f"  Processed {batch_idx*64:,}/{10000:,} samples")


# FINAL RESULTS
test_loss_avg = test_loss / len(test_loader)
cer = compute_cer(all_preds, all_targets)
wer = compute_wer(all_preds, all_targets)


print(f"\n✅ FINAL 10K TEST RESULTS:")
print(f"   Test CTC Loss: {test_loss_avg:.3f}")
print(f"   Test CER: {cer:.1%}")
print(f"   Test WER: {wer:.1%}")
print(f"   Test Samples: {len(all_preds):,}")

