"""
Filename: test_generation.py

Purpose: Test the spatial/character-wise autoregressive generative capacity
         of the trained language NCA model. Given a word prefix, the script
         uses the trained NCA to predict remaining characters up to length 7
         and evaluates if the final generated sequence is a valid English word.

Key Parameters:
 - Grid size: (1, 64, 1, 7) - 1 batch, 64 channels, 1 row, 7 characters max
 - NCA steps per generation: 75 steps per predicted character
 - Target sequence length: Up to 7 characters
 - Evaluation Metric: Exact match against dataset or NLTK corpus (Accuracy %)

Architecture:
 - Must match training exactly:
   - Embedding: 27 classes → 64 dims
   - Conv1 & Conv2: 3x3 kernels (padding=1)
   - Conv3 & Conv4 & Conv5: 1x1 point-wise kernels
   - Final Classifier: 64 dims → 27 classes

Generative Strategy:
 - Start with a prefix (e.g., "a", "ap", "app").
 - Create a fresh grid. Load embedded prefix chars.
 - Leave the current target index as a zero vector.
 - Run NCA for 75 steps. Read prediction.
 - Append predicted character to sequence.
 - Repeat procedure for next target index until EMPTY token or max length 7.
 
Evaluation Strategy:
 - Iterate over every sub-prefix of every word in the original Google 10,000 dataset.
 - Generate complete word from the prefix.
 - Compare against `dataset_set` and NLTK english corpus for validity.

Expected Results:
 - Achieves a high percentage of valid words generated from varied prefixes.
 - Proves NCA architecture can model sequence rules dynamically.

Inputs:
 - letter_predictor.pth: Trained NCA weights

Outputs:
 - Console logs reporting inference progress and final model accuracy.
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import urllib.request
import nltk

# Download standard English words dictionary if not already present
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)
from nltk.corpus import words as nltk_words

# 1. Dataset & Dictionary Setup
url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-no-swears.txt"
response = urllib.request.urlopen(url)
words_list = response.read().decode('utf-8').splitlines()

# The original dataset list
dataset = [w.lower() for w in words_list if 2 <= len(w) <= 7 and w.isalpha()]
dataset_set = set(dataset) # O(1) lookup for dataset
english_dict_set = set(w.lower() for w in nltk_words.words() if w.isalpha()) # O(1) lookup for English

print(f"Loaded {len(dataset)} words from dataset.")
print(f"Loaded {len(english_dict_set)} words from NLTK English dictionary.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using:", device)

EMPTY = 26  # empty/padding class index

def word_to_ints(word):
    """Converts a lowercase string into a list of integer indices (0-25)."""
    return [ord(c) - ord('a') for c in word]

def ints_to_word(ints_list):
    """Converts a list of integer indices (0-25) into a lowercase string."""
    return "".join([chr(i + 97) for i in ints_list])

# 2. Model Architecture (Must match training exactly)
embed_layer = nn.Embedding(27, 64).to(device)
conv1 = nn.Conv2d(64, 512, kernel_size=3, padding=1).to(device)
conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1).to(device)
conv3 = nn.Conv2d(512, 512, kernel_size=1).to(device)
conv4 = nn.Conv2d(512, 512, kernel_size=1).to(device)
conv5 = nn.Conv2d(512, 128, kernel_size=1).to(device)
classifier = nn.Linear(64, 27).to(device)

# 3. Load Weights
checkpoint_path = "letter_predictor.pth"
print(f"Loading weights from {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location=device)

embed_layer.load_state_dict(checkpoint['embed'])
conv1.load_state_dict(checkpoint['conv1'])
conv2.load_state_dict(checkpoint['conv2'])
conv3.load_state_dict(checkpoint['conv3'])
conv4.load_state_dict(checkpoint['conv4'])
conv5.load_state_dict(checkpoint['conv5'])
classifier.load_state_dict(checkpoint['classifier'])

# Set model to evaluation mode
embed_layer.eval()
conv1.eval()
conv2.eval()
conv3.eval()
conv4.eval()
conv5.eval()
classifier.eval()
print("Weights loaded successfully!\n")

# 4. NCA Forward Pass
def step(grid):
    """Performs a single update step of the Neural Cellular Automata using a gated residual block."""
    x = F.leaky_relu(conv1(grid))
    x = F.leaky_relu(conv2(x))
    x = F.leaky_relu(conv3(x))
    x = F.leaky_relu(conv4(x))
    combined = conv5(x)
    update_val, gate = torch.split(combined, 64, dim=1)
    gate = torch.sigmoid(gate)
    return grid + (update_val * gate)

# 5. Generative Inference Function
def generate_from_prefix(prefix):
    """
    Takes a string prefix and autoregressively predicts the remaining characters
    up to a maximum length of 7 using 75 NCA steps per character.
    """
    current_ints = word_to_ints(prefix)
    
    with torch.no_grad():
        # Iterate over the remaining slots to predict up to 7 characters
        for target_idx in range(len(prefix), 7):
            # Fresh grid for each letter generation (matching the training isolation setup)
            grid = torch.zeros(1, 64, 1, 7, device=device)
            
            # Load the known sequence (prefix + generated so far) into the grid
            for pos, char_int in enumerate(current_ints):
                token = torch.tensor([char_int], device=device)
                grid[0, :, 0, pos] = embed_layer(token)
            
            # Run the NCA for 75 steps
            for _ in range(75):
                grid = step(grid)
                
            # Predict the character at target_idx
            pred_vec = grid[0, :, 0, target_idx].unsqueeze(0)
            logits = classifier(pred_vec)
            guess_idx = logits.argmax(dim=1).item()
            
            # If the model predicts the EMPTY token, the word is complete
            if guess_idx == EMPTY:
                break
                
            current_ints.append(guess_idx)
            
    return ints_to_word(current_ints)

# 6. Main Inference Loop
if __name__ == "__main__":
    correct_count = 0
    total_cases = 0
        
    for word_idx, word in enumerate(dataset):
        # Systematically go through all indexes (e.g. "a", "ap", "app", "appl", "apple")
        for i in range(1, len(word) + 1):
            prefix = word[:i]
            total_cases += 1
            
            # Generate word from the current prefix
            generated_word = generate_from_prefix(prefix)
            
            # Check if generated word is in dataset OR the NLTK English dictionary
            is_in_dataset = generated_word in dataset_set
            is_in_english = generated_word in english_dict_set
            
            if is_in_dataset or is_in_english:
                correct_count += 1
                
        # Optional: Log progress every 500 words to show it's alive
        if (word_idx + 1) % 500 == 0:
            accuracy = (correct_count / total_cases) * 100
            print(f"Processed {word_idx + 1}/{len(dataset)} words... Current Accuracy: {accuracy:.2f}% ({correct_count}/{total_cases})")

    # Final Statistics
    print("-" * 60)
    print("Inference Complete!")
    print(f"Total Prefixes Tested: {total_cases}")
    print(f"Valid Words Generated: {correct_count}")
    print(f"Final Model Accuracy:  {(correct_count / total_cases) * 100:.2f}%")