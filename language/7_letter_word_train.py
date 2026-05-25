"""
Filename: 7_letter_word_train.py

Purpose: Train a Neural Cellular Automata (NCA) to act as a character-level
         autoregressive language model. The NCA learns the structure (phonotactics)
         of English words (up to 7 letters) by observing random sub-word states 
         and predicting the missing masked character.

Key Parameters:
 - Grid size: (1, 64, 1, 7) - 1 batch, 64 channels, 1 row, 7 characters max
 - Learning rate: 0.0001 (Cosine Annealing scheduler)
 - NCA steps per token prediction: 75
 - Training iterations: 400,000
 - Input: Characters embedded as 64-dimensional vectors. Masked index is a zero vector.
 - Loss: CrossEntropyLoss on the masked character position

Architecture:
 - Embed Layer: 27 classes (a-z + <empty>) mapped to 64 dimensions.
 - Conv1: 64→512 channels, kernel=(3,3), padding=(1,1) (Local context mixing)
 - Conv2: 512→512 channels, kernel=(3,3), padding=(1,1)
 - Conv3 & Conv4: 512→512 channels, kernel=(1,1) (Point-wise processing)
 - Conv5: 512→128 channels, kernel=(1,1) (Outputs 64 channel update, 64 channel gate)
 - Residual Update: Gated residual connection typical for NCAs
 - Classifier: Linear(64, 27) (Predicts class from final state of masked channel)
 - Activation: LeakyReLU

Rules Being Learned:
 - English vocabulary phonotactics and typical character sequences
 - Given a partial valid English string, predict the most likely character in the missing slot
 - NCA discovers these sequence properties purely from the text dataset

Training Strategy:
 - Randomly sample a word from a 10,000 common English word dataset (2-7 characters).
 - Pick a random index to "eject" (mask out as a zero vector).
 - If word is shorter than 7, remaining padded slots use <empty> class mapping.
 - Run 75 NCA communication steps.
 - Apply CrossEntropy to final cell vector prediction at the ejected index vs the true character.

Expected Results:
 - Loss converges over time.
 - NCA gains ability to accurately guess plausible missing English characters.

Outputs:
 - letter_predictor.pth: Checkpoint containing 'embed', 'conv1-5', 'classifier', 'optimizer', and 'scheduler' state dicts
"""
import torch
import torch.nn.functional as F
import random
import torch.nn as nn
import urllib.request

# 1. Dataset setup
url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-no-swears.txt"
response = urllib.request.urlopen(url)
words = response.read().decode('utf-8').splitlines()
dataset = [w.lower() for w in words if 2 <= len(w) <= 7 and w.isalpha()]

print(f"Total words in your dataset: {len(dataset)}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using:", device)

EMPTY = 26  # empty/padding class index

def word_to_ints(word):
    """Converts a lowercase string into a list of integer indices (0-25)."""
    return [ord(c) - ord('a') for c in word]

def idx_to_char(i):
    """Converts an integer index back to a character, or '*' for the EMPTY token."""
    return '*' if i == EMPTY else chr(i + 97)

# 2. Model Architecture
embed_layer = nn.Embedding(27, 64).to(device)  # 27: a-z + empty
conv1 = nn.Conv2d(64, 512, kernel_size=3, padding=1).to(device)
conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1).to(device)
conv3 = nn.Conv2d(512, 512, kernel_size=1).to(device)
conv4 = nn.Conv2d(512, 512, kernel_size=1).to(device)
conv5 = nn.Conv2d(512, 128, kernel_size=1).to(device)
classifier = nn.Linear(64, 27).to(device)  # 27 output classes

nn.init.zeros_(conv5.weight)
nn.init.zeros_(conv5.bias)

optimizer = torch.optim.AdamW(
    list(embed_layer.parameters()) +
    list(conv1.parameters()) + list(conv2.parameters()) +
    list(conv3.parameters()) + list(conv4.parameters()) +
    list(conv5.parameters()) + list(classifier.parameters()),
    lr=0.0001, weight_decay=0.01
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=400000, eta_min=1e-6
)

def save_weights(iteration, filename):
    """Saves model weights, optimizer, and scheduler states for checkpoints."""
    checkpoint = {
        'iteration': iteration,
        'embed': embed_layer.state_dict(),
        'conv1': conv1.state_dict(),
        'conv2': conv2.state_dict(),
        'conv3': conv3.state_dict(),
        'conv4': conv4.state_dict(),
        'conv5': conv5.state_dict(),
        'classifier': classifier.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(checkpoint, filename)
    print(f"💾 Saved checkpoint at iteration {iteration}")

# 4. Forward pass
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

# 5. Training step
def trainingLoop(log, word, iteration):
    """
    Executes one training iteration. Masks one character in the target word,
    initializes the NCA grid, runs 75 update steps, and computes the loss
    on the missing character prediction.
    """
    word_ints = word_to_ints(word)
    word_len = len(word_ints)

    ejected_idx = random.randint(0, 6)

    # Target: letter at that position, or EMPTY if past the end of the word
    target_class = word_ints[ejected_idx] if ejected_idx < word_len else EMPTY
    target = torch.tensor([target_class], device=device)

    # Build grid — fill positions 0..ejected_idx-1, leave ejected_idx as zero vector
    grid = torch.zeros(1, 64, 1, 7, device=device)
    with torch.no_grad():
        for pos in range(ejected_idx):
            if pos < word_len:
                # Real letter from the word
                token = torch.tensor([word_ints[pos]], device=device)
            else:
                # Word ended before ejected_idx — fill with EMPTY embedding
                token = torch.tensor([EMPTY], device=device)
            grid[0, :, 0, pos] = embed_layer(token)
        # ejected_idx slot stays as zero vector (the blank the model predicts)

    optimizer.zero_grad()
    losses = []

    for _ in range(75):
        grid = step(grid)
        pred_vec = grid[0, :, 0, ejected_idx]
        logits = classifier(pred_vec.unsqueeze(0))
        step_loss = F.cross_entropy(logits, target)
        losses.append(step_loss)

    loss = torch.stack(losses).mean()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if log:
        with torch.no_grad():
            pred_vec = grid[0, :, 0, ejected_idx].unsqueeze(0)
            logits = classifier(pred_vec)
            probs = F.softmax(logits, dim=1)
            guess_idx = probs.argmax(dim=1).item()

            # Build display string: filled chars, then blank at ejected_idx
            display = []
            for pos in range(ejected_idx + 1):
                if pos == ejected_idx:
                    display.append('_')
                elif pos < word_len:
                    display.append(chr(word_ints[pos] + 97))
                else:
                    display.append('*')
            display_str = ''.join(display)

            top5_probs, top5_idx = torch.topk(probs[0], 5)
            top5_str = " ".join(
                f"{idx_to_char(top5_idx[k].item())}:{top5_probs[k].item()*100:.1f}%"
                for k in range(5)
            )

            print(
                f"Iter {iteration} | {display_str} -> {idx_to_char(target_class)} | "
                f"Guess: {idx_to_char(guess_idx)} | Loss: {loss.item():.6f} | Top5: {top5_str}"
            )

# 6. Main
if __name__ == "__main__":
    for i in range(400000):
        word = dataset[random.randint(0, len(dataset) - 1)]
        trainingLoop(i % 1000 == 0, word, i)

        if i > 0 and i % 5000 == 0:
            save_weights(i, filename="letter_predictor.pth")

    save_weights(400000, filename="letter_predictor.pth")
    print("Training complete!")