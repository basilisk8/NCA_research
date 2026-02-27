"""
Filename: gated_residual_test.py

Purpose: Test whether trained NCA generalizes to unseen data.
         Loads weights from training on width 7 and test on width 7 width 5 and width 10

Requirements:
 - gated_residual_sort.pth (from training script with width 7)

Test Strategy:
 - Training data Width 7 : 100 random combinations
 - Unseen data Width 5 and width 10: 20 random combinations

Expected Results:
 - Training data: 99% accuracy (99/100)
 - Unseen data Width 5 : 100% accuracy (20/20)
 - Unseen data Width 10 : 85% accuracy (17/20)
 - Model learned sorting, the low accuracy on width 10 could be because of not enough steps

Output Format:
 - Prints accuracy on all 3 testings
 - Prints close values in width 4, failures on width 10 and width 7
"""

import torch
import torch.nn as nn
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

conv = nn.Conv2d(64, 126, kernel_size=(3, 3), padding=(1, 1)).to(device)
conv.load_state_dict(torch.load('gated_residual_sort.pth'))
conv.eval()

def step(grid):
    raw_update = conv(grid)
    v = raw_update[:, 0::2, :, :]
    g = raw_update[:, 1::2, :, :]
    gated_update = v * torch.sigmoid(g)
    newGrid = grid.clone()
    newGrid[0, 1:, :, :] = grid[0, 1:, :, :] + gated_update[0, :, :, :]
    return newGrid


def test_sort(values):
    width = len(values)
    grid = torch.zeros(1, 64, 1, width).to(device)
    for i in range(width):
        grid[0, 0, 0, i] = values[i]

    for _ in range(60):
        grid = step(grid)

    output = grid[0, 1, 0, :width]
    return [round(x.item()) for x in output]


def is_correct(pred, expected, tolerance=5):
    # Check ordering
    for i in range(len(pred) - 1):
        if pred[i] > pred[i + 1]:
            return False
    # Check values within tolerance
    for p, e in zip(pred, expected):
        if abs(p - e) > tolerance:
            return False
    return True


print("=" * 60)
print("TEST 1: 100 random width-7 cases")
print("=" * 60)

random.seed(42)
exact = 0
tolerance_correct = 0

for _ in range(100):
    values = [random.randint(0, 1) * random.randint(0, 255) for _ in range(7)]
    expected = sorted(values)
    pred = test_sort(values)

    if pred == expected:
        exact += 1
        tolerance_correct += 1
    elif is_correct(pred, expected, tolerance=5):
        tolerance_correct += 1
        print(f"  CLOSE: {values} → {pred} expected {expected}")
    else:
        print(f"  FAIL: {values} → {pred} expected {expected}")

print(f"\n  Exact: {exact}/100")
print(f"  Within ±5 and ordered: {tolerance_correct}/100")


print("\n" + "=" * 60)
print("TEST 2: Width-5 (never trained)")
print("=" * 60)

correct_5 = 0
for _ in range(20):
    values = [random.randint(0, 1) * random.randint(0, 255) for _ in range(5)]
    expected = sorted(values)
    pred = test_sort(values)
    if is_correct(pred, expected, tolerance=5):
        correct_5 += 1
    else:
        print(f"  FAIL: {values} → {pred} expected {expected}")

print(f"  Accuracy: {correct_5}/20")


print("\n" + "=" * 60)
print("TEST 3: Width-10 (never trained)")
print("=" * 60)

correct_10 = 0
for _ in range(20):
    values = [random.randint(0, 1) * random.randint(0, 255) for _ in range(10)]
    expected = sorted(values)
    pred = test_sort(values)
    if is_correct(pred, expected, tolerance=5):
        correct_10 += 1
    else:
        print(f"  FAIL: {values} → {pred} expected {expected}")

print(f"  Accuracy: {correct_10}/20")