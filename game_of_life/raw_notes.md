NCA learns local rules, as shown by maxwell and heat diffusion results, so then NCA should in theory be able to learn the local rules for GoL. The idea is simple, train it on given step n get to step n + 1, then test generalization across time and space.

during training
Using: cuda
Iter 0 | loss=0.755672
Iter 5000 | loss=0.017379
Iter 10000 | loss=0.000820
Iter 15000 | loss=0.000047
Iter 20000 | loss=0.000003
Iter 25000 | loss=0.000000
Iter 30000 | loss=0.000000
Iter 35000 | loss=0.000000
Iter 40000 | loss=0.000000
Iter 45000 | loss=0.000000

during generalization testing
==================================================
SPATIAL GENERALIZATION
==================================================
  16x16: 20/20 perfect
  32x32: 20/20 perfect
  64x64: 20/20 perfect
  128x128: 20/20 perfect
  256x256: 20/20 perfect

==================================================
TIME GENERALIZATION (autoregressive)
==================================================
  10 steps: 100.0% avg final accuracy
  20 steps: 100.0% avg final accuracy
  50 steps: 100.0% avg final accuracy
  100 steps: 100.0% avg final accuracy

==================================================
TIME GENERALIZATION (step by step detail)
==================================================
  Step   1: 100.0%
  Step   2: 100.0%
  Step   3: 100.0%
  Step   4: 100.0%
  Step   5: 100.0%
  Step   6: 100.0%
  Step   7: 100.0%
  Step   8: 100.0%
  Step   9: 100.0%
  Step  10: 100.0%
  Step  15: 100.0%
  Step  20: 100.0%
  Step  25: 100.0%
  Step  30: 100.0%
  Step  35: 100.0%
  Step  40: 100.0%
  Step  45: 100.0%
  Step  50: 100.0%

==================================================
TIME + SPACE (large grid, many steps)
==================================================
  64x64 after 50 steps: 100.0%
  128x128 after 50 steps: 100.0%
