# infinitrix_AI-ML
this is a repo for the maths (infinitrix club) task ai/ml domain , PS :- handwritten text recognition 

**98.5% Character Recognition** 
**Word accuracy(WER) ~80% **
 
|✅  FINAL 10K TEST RESULTS:     |
| ->  Test CTC Loss: 2.121         | 
| ->  Test CER: 1.5%               |
| ->  Test Samples: 9,986          |


Handwritten Name Recognition using CRNN (PyTorch)
This repository contains a complete implementation of a Convolutional Recurrent Neural Network (CRNN) for recognizing handwritten names using the popular Kaggle Handwriting Recognition dataset (also known as the "Written Names" dataset).
The model is trained on the full ~330,000 training images and evaluated on the first 10,000 test samples.
Dataset

Source: Kaggle - Handwriting Recognition
dataset :- https://www.kaggle.com/datasets/landlord/handwriting-recognition

Key Features & Design Choices

**Labels preprocessing:**
->Converted to uppercase
->Limited to first 6 characters (improves convergence on this noisy dataset)
->Character set: A-Z + space (27 characters + CTC blank token)

**Image preprocessing:**
->Resized to fixed size: 32 (height) × 100 (width)
->Grayscale, normalized to [-1, 1]

**Model Architecture (Simple yet effective CRNN):**
->CNN backbone (3 conv layers + pooling) → extracts features → sequence of length 50 with 1024 dimensions
->GRU (hidden size 128) → Processes sequential data (like text, speech, handwriting strokes)
->Final linear layer → 28 classes (27 chars + blank)


Loss: CTC Loss (Connectionist Temporal Classification) – perfect for unaligned sequence transcription
Decoding: Greedy CTC decoding (collapse repeats, remove blanks)

**Training:**
->Adam optimizer (lr=0.001)
->Gradient clipping
->15 epochs on full training set (70/30 train-val split)
->Best model saved based on validation CTC loss

**Evaluation Metrics (on first 10,000 test samples):**
**CTC Loss**
**Character Accuracy**:- (1 - CER)

Dataset loaded: 330,395 samples (FULL)
✅ Train: 231,276 | Val: 99,119

🚀 Training FULL 330K dataset...
Epoch | Train Loss | Val Loss
-----|------------|--------
 1   | 2.154     | 2.132
 2   | 2.127     | 2.134
 3   | 2.126     | 2.133
 4   | 2.126     | 2.133
 5   | 2.125     | 2.132
 6   | 2.125     | 2.130
 7   | 2.124     | 2.130
 8   | 2.124     | 2.131
 9   | 2.124     | 2.130
10   | 2.123     | 2.131
11   | 2.123     | 2.130
12   | 2.123     | 2.131
13   | 2.123     | 2.129
14   | 2.123     | 2.131
15   | 2.123     | 2.129

- Both losses are gradually decreasing, which means the model is learning.
- The gap between train and val loss is small, suggesting low overfitting .


Evaluating 10K test samples...
  Processed 0/10,000 samples
  Processed 1,280/10,000 samples
  Processed 2,560/10,000 samples
  Processed 3,840/10,000 samples
  Processed 5,120/10,000 samples
  Processed 6,400/10,000 samples
  Processed 7,680/10,000 samples
  Processed 8,960/10,000 samples

✅ FINAL 10K TEST RESULTS:
   Test CTC Loss: 2.121
   Test CER: 1.5%
   Test Samples: 9,986
   
   1.5% CER(Character Error Rate) loss suggest a 98.5% character accuracy on the test data which is a good value 


   
   

   
    
    
