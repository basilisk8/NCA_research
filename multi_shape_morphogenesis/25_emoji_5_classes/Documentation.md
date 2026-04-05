# 25 Emoji Factored Embeddings NCA

## Overview
This experiment pushes the limits of NCA conditional image generation by teaching a single set of weights to grow 25 different RGB images. The targets are arranged in a 5×5 grid representing 5 semantic classes (different emoji subjects) across 5 variants (different styles or platforms). Rather than using 25 independent embeddings, the architecture factors the condition into two separate learned embeddings: a class embedding and a variant embedding, forcing the NCA to disentangle "content" from "style".

## Key Question

**Can a single NCA learn to generate a matrix of complex RGB targets (5 classes × 5 variants) by using factored seed embeddings, effectively disentangling content from style?**

## Architecture

- **Grid**: `(1, 192, 40, 40)` — 1 batch, 192 channels, 40×40 cells
  - Channels 0-2: RGB visible output (compared to target)
  - Channels 3-127: Hidden state for growth computation (125 channels)
  - Channels 128-159: Class embedding (32 channels, constant)
  - Channels 160-191: Variant embedding (32 channels, constant)
- **Seed**: 
  - `class_embed = nn.Embedding(5, 32)`
  - `variant_embed = nn.Embedding(5, 32)`
  - Concatenated to form a 64-channel seed placed at the center cell (20, 20)
- **Convolution 1**: 192→256 channels, 3×3 kernel, padding=1, ReLU
- **Convolution 2**: 256→256 channels, 1×1 kernel, ReLU
- **Convolution 3**: 256→128 channels, 1×1 kernel, zero-initialized, NO activation 
- **Stochastic mask**: **REMOVED**. Standard stochastic continuous masking caused outputs to blur.
- **Update scaling**: Multiply by 0.1
- **Steps**: Random 128-160 per training example
- **Loss**: MSELoss on channels 0-2 (RGB) vs target emoji
- **Gradient clipping**: Max norm 1.0
- **Optimizer**: AdamW, lr=0.0005, weight_decay=0.01

## Files

### Training Scripts

**`train_25_emoji.py`**
- Main training script for the factored embedding architecture.
- Randomly samples a class and variant, generating the specific target.
- Runs for 200,000 iterations without a stochastic mask.
- Computes loss across a random step span of 128 to 160 steps to encourage temporal stability.
- Outputs weights to `factored_256ch_nomask.pth`.

### Testing Scripts

**`test_25_emoji.py`**
- Loads the trained factored weights.
- Iterates over all 25 class/variant combinations, running the NCA for 130 steps.
- Computes loss and generates a grid visualization (`test_results.png`) comparing the true targets against the generated outputs.

## Experiments & Results

### Failed Attempts

Several architectures and parameters were explored that completely failed to capture the complexity of 25 unique items:
- **5 Flat Embeddings (16-20 dim, 64 hidden, 128 conv)**: The loss plateaued between 0.02 and 0.06. The NCA optimized by predicting the literal blurry average of all 5 variant superimpositions for a given class. It proved the model had no capacity to separate the targets at those dimensionalities.
- **4 Convolutions instead of 3**: Failed due to training instability, resulting in stagnant losses and much slower iterations.
- **High Steps (128-192) on Old Architecture**: Adding more steps to the older configurations only degraded the output and dramatically increased training time.
- **Early Factored Variants (Smaller channels + Stochastic Mask)**: An early attempt at separating into `class` and `variant` embeddings (diffusion-model style) dropped the loss to 0.001 but yielded blurry grown outputs. This proved the stochastic mask inhibited the sharp formation of highly detailed color variations.

### Successful Attempt (Factored + Wide Channels + No Mask)

**Setup**: Increased hidden channels to 128 state and 256 conv, used the factored embeddings (32 dims each), removed the stochastic mask, and trained for 200k iterations.

**Results**:
- The model successfully learned to disentangle the class and variant, growing the correct combination with distinct shapes and styles.
- Removing the stochastic mask in continuous color channels combined with the larger capacity network allowed the NCA to represent high-frequency visual details sharply.
- **Overfitting Dynamics**: It was observed that weights from 200k iterations performed visually better than those trained to 300k iterations. Because of `AdamW`'s weight decay fighting against the gradients after long continuous plateaus, the weights eventually began shrinking toward 0 excessively, causing the NCA to lose stability and degrade in output quality at higher iteration counts. The 200k mark achieved optimal stability and visual fidelity.

## Future Directions
- **Expanding to imagenet-scale targets**: Testing whether this factored embedding approach can scale to more complex datasets with hundreds of classes and variants.
- **Dynamic Conditioning**: Exploring whether the class and variant embeddings can be dynamically updated during growth rather than being static, allowing for more flexible generation.
- **Using scheduler for steps**: Implementing a learning rate scheduler or step scheduler to optimize training dynamics and prevent overfitting at later iterations.