Pre experiment
 - Is sequencial part the only non parrelal part in NCA or is cell update also sequential?
 - If cell update is parrelal then as long as the grid size scales reasonable within the PGU limits, the time taken should not change
 - Test training time to train a model that is 11 wide vs 30 wide

result :
Using: cuda
Training on grid size 11
Logit range: [-12.06, 3.36]
Iteration 0 | 39+18=57 | Loss: 1.881581
Logit range: [-8.23, 2.84]
Iteration 1000 | 10+75=85 | Loss: 0.673211
Logit range: [-8.59, 3.31]
Iteration 2000 | 91+84=175 | Loss: 0.273946
Logit range: [-8.54, 5.43]
Iteration 3000 | 41+75=116 | Loss: 0.449143
Logit range: [-9.31, 5.85]
Iteration 4000 | 7+36=43 | Loss: 0.027938
Logit range: [-9.49, 5.64]
Iteration 5000 | 43+38=81 | Loss: 0.022600
Logit range: [-9.90, 7.94]
Iteration 6000 | 12+32=44 | Loss: 0.002462
Logit range: [-9.18, 6.14]
Iteration 7000 | 52+75=127 | Loss: 0.303386
Logit range: [-9.28, 7.94]
Iteration 8000 | 92+67=159 | Loss: 0.003360
Logit range: [-9.69, 6.32]
Iteration 9000 | 49+47=96 | Loss: 0.212666
Time taken for grid size 11 is : 112.02447938919067
Training on grid size 30
Logit range: [-14.91, 10.02]
Iteration 0 | 15+28=43 | Loss: 7.690719
Logit range: [-12.07, 0.43]
Iteration 1000 | 9+15=24 | Loss: 0.108623
Logit range: [-10.52, 1.73]
Iteration 2000 | 51+20=71 | Loss: 0.149359
Logit range: [-9.96, 2.60]
Iteration 3000 | 66+52=118 | Loss: 0.090482
Logit range: [-9.11, 3.01]
Iteration 4000 | 15+30=45 | Loss: 0.144006
Logit range: [-10.49, 3.95]
Iteration 5000 | 62+73=135 | Loss: 0.156407
Logit range: [-11.15, 5.06]
Iteration 6000 | 3+98=101 | Loss: 0.005216
Logit range: [-10.96, 5.47]
Iteration 7000 | 36+62=98 | Loss: 0.007248
Logit range: [-11.17, 6.15]
Iteration 8000 | 66+63=129 | Loss: 0.001713
Logit range: [-11.65, 7.00]
Iteration 9000 | 96+56=152 | Loss: 0.000470
TIme taken for gird size 30 is 104.1686692237854

Conclusion 
 - loss is about the same
 - grid being wider doesn't change training time but 30 wide grid can store a lot more values than a 11 wide grid
 - theoretically you can just make a 100 wide and then compute very very very big numbers without it affecting the training time
