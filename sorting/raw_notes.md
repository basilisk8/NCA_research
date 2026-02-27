Pure sorting fails when I used addition architecture 
 - Values aren't preserved
 - Weights can't be found that swap numbers without destroying their values

Iter 0 | [144,236]→[-278.6,204.6] expected [236,144] | Swap: 2.064111 | Preserve: 1.380546 | Loss: 140.118744
Iter 5000 | [130,220]→[240.0,118.6] expected [220,130] | Swap: 0.004061 | Preserve: 0.004061 | Loss: 0.410198
Iter 10000 | [174,55]→[41.3,182.9] expected [55,174] | Swap: 0.002045 | Preserve: 0.002045 | Loss: 0.206579
Iter 15000 | [51,226]→[54.9,214.8] expected [226,51] | Swap: 0.431288 | Preserve: 0.001086 | Loss: 0.539887
Iter 20000 | [222,182]→[221.6,182.4] expected [182,222] | Swap: 0.024067 | Preserve: 0.000003 | Loss: 0.024365
Iter 25000 | [224,236]→[235.5,224.4] expected [236,224] | Swap: 0.000003 | Preserve: 0.000003 | Loss: 0.000293
Iter 30000 | [210,153]→[153.0,210.2] expected [153,210] | Swap: 0.000000 | Preserve: 0.000000 | Loss: 0.000031
Iter 35000 | [81,142]→[141.8,80.9] expected [142,81] | Swap: 0.000000 | Preserve: 0.000000 | Loss: 0.000035
Iter 40000 | [87,122]→[122.0,86.7] expected [122,87] | Swap: 0.000001 | Preserve: 0.000001 | Loss: 0.000067

swap and preserve works. First they were same losses, then it learnt preservation is more important than swap, then when preservation loss was near 0, it then decreased swap to decrease total loss

NCA can compute when to swap, but can not sort for some reason?

Iter 220000 | [231,240]→[233,240] expected [231,240] | ✗ | Loss: 0.000027
Iter 230000 | [158,234]→[158,234] expected [158,234] | ✓ | Loss: 0.000002
Iter 240000 | [65,161]→[67,159] expected [65,161] | ✗ | Loss: 0.000082
Iter 250000 | [145,9]→[10,145] expected [9,145] | ✗ | Loss: 0.000003
Iter 260000 | [109,14]→[18,106] expected [14,109] | ✗ | Loss: 0.000181
Iter 270000 | [213,60]→[60,214] expected [60,213] | ✗ | Loss: 0.000008
Iter 280000 | [159,146]→[145,161] expected [146,159] | ✗ | Loss: 0.000028
Iter 290000 | [13,197]→[15,194] expected [13,197] | ✗ | Loss: 0.000082
Iter 300000 | [160,69]→[66,156] expected [69,160] | ✗ | Loss: 0.000169
Iter 310000 | [205,91]→[92,205] expected [91,205] | ✗ | Loss: 0.000010
Iter 320000 | [237,77]→[77,237] expected [77,237] | ✓ | Loss: 0.000001
Iter 330000 | [104,170]→[105,170] expected [104,170] | ✗ | Loss: 0.000004
Iter 340000 | [172,92]→[91,172] expected [92,172] | ✗ | Loss: 0.000005
Iter 350000 | [89,141]→[90,140] expected [89,141] | ✗ | Loss: 0.000009
Iter 360000 | [219,122]→[124,217] expected [122,219] | ✗ | Loss: 0.000046
Iter 370000 | [242,154]→[154,242] expected [154,242] | ✓ | Loss: 0.000001
Iter 380000 | [78,209]→[78,209] expected [78,209] | ✓ | Loss: 0.000001
Iter 390000 | [191,51]→[52,190] expected [51,191] | ✗ | Loss: 0.000014
Iter 400000 | [190,72]→[72,189] expected [72,190] | ✗ | Loss: 0.000008
Iter 410000 | [160,140]→[141,159] expected [140,160] | ✗ | Loss: 0.000014
Iter 420000 | [98,151]→[98,152] expected [98,151] | ✗ | Loss: 0.000006
Iter 430000 | [145,251]→[146,251] expected [145,251] | ✗ | Loss: 0.000003
Iter 440000 | [213,109]→[110,212] expected [109,213] | ✗ | Loss: 0.000012

proves NCA can actually sort 2 numbers, just takes about 500k itterations and 32 channels with 30 steps. this training is painfully slow, can this be done faster than 2 hours? The NCA did learn sorting though, but not always point on because of tanh's continues domain it is off by 0.5, 

raw outputs
INPUT           | NCA OUTPUT (Raw)     | ROUNDED         | EXPECTED        | LOGIC
------------------------------------------------------------------------------------------
[182, 241]      | [182.8, 241.1]      | [183, 241]     | [182, 241]     | ✓ (Off by 1)
[167, 142]      | [143.1, 166.1]      | [143, 166]     | [142, 167]     | ✓ (Off by 2)
[111, 205]      | [111.5, 204.8]      | [112, 205]     | [111, 205]     | ✓ (Off by 1)
[64, 214]       | [64.1, 214.4]       | [64, 214]      | [64, 214]      | ✓ Perfect
[112, 99]       | [100.3, 110.9]      | [100, 111]     | [99, 112]      | ✓ (Off by 2)
[12, 225]       | [12.0, 225.6]       | [12, 226]      | [12, 225]      | ✓ (Off by 1)
[236, 113]      | [113.1, 235.4]      | [113, 235]     | [113, 236]     | ✓ (Off by 1)
[167, 172]      | [166.9, 172.4]      | [167, 172]     | [167, 172]     | ✓ Perfect
[152, 147]      | [146.7, 152.5]      | [147, 152]     | [147, 152]     | ✓ Perfect
[87, 138]       | [87.7, 138.0]       | [88, 138]      | [87, 138]      | ✓ (Off by 1)
[207, 159]      | [158.5, 207.8]      | [159, 208]     | [159, 207]     | ✓ (Off by 1)
[17, 67]        | [17.5, 67.0]        | [17, 67]       | [17, 67]       | ✓ Perfect
[158, 221]      | [158.6, 221.1]      | [159, 221]     | [158, 221]     | ✓ (Off by 1)
[226, 149]      | [148.5, 226.4]      | [148, 226]     | [149, 226]     | ✓ (Off by 1)
[128, 74]       | [73.5, 128.7]       | [73, 129]      | [74, 128]      | ✓ (Off by 2)
[187, 253]      | [187.7, 253.1]      | [188, 253]     | [187, 253]     | ✓ (Off by 1)
[193, 154]      | [154.0, 193.3]      | [154, 193]     | [154, 193]     | ✓ Perfect
[181, 235]      | [181.9, 235.0]      | [182, 235]     | [181, 235]     | ✓ (Off by 1)
[200, 215]      | [202.1, 213.5]      | [202, 213]     | [200, 215]     | ✓ (Off by 4)
[74, 232]       | [74.1, 232.6]       | [74, 233]      | [74, 232]      | ✓ (Off by 1)
is it possible to be more presise?

NCA absolutely fails at generalization of this
probably because it has never seen how to compute more than 3 digits because it never encountered 'middle' digit in trainings
can reLU do this better? maybe not because reLU can't subtract.

The train_compare_and_swap architecure doesn't scale beyond 2 numbers because mathematically interpolation is the lowest efforst ofr lowest loss possible

what if we encode 1 binary digit in 1 channel, and try to compute with that architecure
the architecture will be unstable though because even NCA's locaility has limits on info propogation, especially since information is spaced out in channels

what if we just punish the NCA through loss that interpolation is actually not the methematically lowest loss for lowest effort
that didn't work, probably because gradient descent found the local minimum to be the identity function, and even adaptive loss doesn't fix that

The core problem here is that preservation loss and sorting loss contradict each other, to sort 4 elements the NCA has to try combinations of sorting, and trying to sort heavily punishes because sorting loss is high and preservation loss is high. So we need to reward trying to swap

What if we use cross entrophy. the cells have to 'rank' themself instead of having to preserve and swap and compare. so input of [3 7 0 1] expects the output [3 4 1 2]
This works, but the model get's values that are close always wrong. probably because the division by 255 looses some accuracy as now numbers that are close are like 0.01 away because The division
This architecture can not generalize though. like cross entrophy needs 1 output channel for every tpye of classification, so 4 element needs 4 channels but 8 elements needs 8 channels

what if the NCA outputs the ranks in 1 channel, like input of [3 7 0 1] gives output of [3 4 1 2] in 1 channel. so training on variable grid size should force NCA to generalize to grid sizes never seen before. 
This failed badly. because the difference in decimal point for numbers close is too close, ranking fails in a typical MSE loss

NCA axoim says NCA's can find rules for local functions. it did find the rule for the local function with corss entrophy. now in some cases like heat diffusion, this can scale and generalize, but that's a byproduct of the axoim and not guarenteed. so NCA can find the local rule for sorting. 

But bubble sort is local and it works, so there is no reason for this to not work. hungarian loss could potentially work better than MSE preservation loss, but the core issue is still that 1 NCA might not be able to compare, swap, and preserve values across multiple steps, so maybe we just need to add more NCA's, similar to how 1 NN layer couldn't do XOR, but you just add more layers and now it can.

that kinda works, 

Iter 370000 | Loss: 0.215168 | ✗
  Input:    [90, 98, 56, 22]
  Expected: [22, 56, 90, 98]
  Got:      [28, 59, 87, 96]
  Sort Loss: 0.0034 | Match Loss: 0.2118

Iter 375000 | Loss: 0.230443 | ✗
  Input:    [236, 37, 120, 208]
  Expected: [37, 120, 208, 236]
  Got:      [29, 122, 209, 239]
  Sort Loss: 0.0051 | Match Loss: 0.2254

Iter 380000 | Loss: 0.221857 | ✗
  Input:    [200, 84, 42, 231]
  Expected: [42, 84, 200, 231]
  Got:      [41, 88, 191, 231]
  Sort Loss: 0.0057 | Match Loss: 0.2162

Iter 385000 | Loss: 0.094744 | ✗
  Input:    [23, 88, 119, 127]
  Expected: [23, 88, 119, 127]
  Got:      [27, 88, 117, 127]
  Sort Loss: 0.0011 | Match Loss: 0.0937

Iter 390000 | Loss: 0.123931 | ✗
  Input:    [174, 188, 193, 37]
  Expected: [37, 174, 188, 193]
  Got:      [38, 174, 184, 196]
  Sort Loss: 0.0013 | Match Loss: 0.1226

Iter 395000 | Loss: 0.081320 | ✗
  Input:    [121, 82, 46, 158]
  Expected: [46, 82, 121, 158]
  Got:      [46, 85, 122, 157]
  Sort Loss: 0.0006 | Match Loss: 0.0807

Iter 400000 | Loss: 0.221305 | ✗
  Input:    [179, 228, 203, 2]
  Expected: [2, 179, 203, 228]
  Got:      [2, 179, 209, 220]
  Sort Loss: 0.0058 | Match Loss: 0.2155

Iter 405000 | Loss: 0.159932 | ✗
  Input:    [218, 13, 217, 125]
  Expected: [13, 125, 217, 218]
  Got:      [9, 125, 218, 223]
  Sort Loss: 0.0026 | Match Loss: 0.1573

Iter 410000 | Loss: 0.205893 | ✗
  Input:    [218, 60, 59, 116]
  Expected: [59, 60, 116, 218]
  Got:      [57, 59, 121, 213]
  Sort Loss: 0.0034 | Match Loss: 0.2025

Iter 415000 | Loss: 0.030483 | ✗
  Input:    [37, 65, 235, 176]
  Expected: [37, 65, 176, 235]
  Got:      [38, 64, 176, 235]
  Sort Loss: 0.0001 | Match Loss: 0.0304

Iter 420000 | Loss: 0.229606 | ✗
  Input:    [15, 214, 195, 105]
  Expected: [15, 105, 195, 214]
  Got:      [16, 110, 190, 218]
  Sort Loss: 0.0038 | Match Loss: 0.2258

but when testing the weights
Using: cuda
============================================================
TEST 1: 100 random width-4 cases
============================================================

  FAIL: [235, 1, 185, 193] → [8, 181, 204, 226] expected [1, 185, 193, 235]
  FAIL: [193, 235, 216, 199] → [189, 205, 216, 237] expected [193, 199, 216, 235]
  FAIL: [6, 73, 5, 203] → [0, 15, 69, 203] expected [5, 6, 73, 203]
  FAIL: [113, 8, 55, 49] → [7, 43, 61, 111] expected [8, 49, 55, 113]
  FAIL: [12, 226, 146, 148] → [17, 141, 156, 221] expected [12, 146, 148, 226]
  FAIL: [129, 128, 211, 252] → [119, 137, 214, 253] expected [128, 129, 211, 252]
  FAIL: [25, 13, 11, 140] → [3, 16, 30, 138] expected [11, 13, 25, 140]
  FAIL: [12, 210, 0, 71] → [-3, 19, 66, 212] expected [0, 12, 71, 210]
  FAIL: [152, 195, 6, 152] → [7, 148, 161, 191] expected [6, 152, 152, 195]
  FAIL: [253, 162, 187, 86] → [86, 162, 194, 248] expected [86, 162, 187, 253]
  FAIL: [129, 9, 7, 214] → [-2, 19, 130, 214] expected [7, 9, 129, 214]
  FAIL: [137, 85, 221, 130] → [83, 125, 145, 221] expected [85, 130, 137, 221]
  FAIL: [123, 199, 139, 251] → [117, 144, 201, 254] expected [123, 139, 199, 251]
  FAIL: [43, 134, 18, 167] → [14, 49, 131, 168] expected [18, 43, 134, 167]
  FAIL: [224, 137, 90, 90] → [81, 95, 141, 225] expected [90, 90, 137, 224]
  FAIL: [139, 165, 89, 55] → [49, 91, 140, 169] expected [55, 89, 139, 165]
  FAIL: [213, 7, 0, 135] → [-4, 15, 131, 214] expected [0, 7, 135, 213]
  FAIL: [159, 46, 1, 195] → [-3, 52, 159, 193] expected [1, 46, 159, 195]
  FAIL: [9, 147, 250, 221] → [16, 144, 222, 248] expected [9, 147, 221, 250]
  FAIL: [112, 22, 24, 255] → [14, 37, 115, 251] expected [22, 24, 112, 255]
  FAIL: [108, 216, 156, 105] → [97, 117, 156, 217] expected [105, 108, 156, 216]
  FAIL: [222, 26, 59, 228] → [27, 64, 214, 233] expected [26, 59, 222, 228]
  FAIL: [147, 184, 220, 2] → [3, 144, 190, 219] expected [2, 147, 184, 220]
  FAIL: [49, 193, 1, 246] → [2, 57, 189, 243] expected [1, 49, 193, 246]
  FAIL: [174, 98, 146, 46] → [44, 97, 152, 171] expected [46, 98, 146, 174]
  FAIL: [219, 16, 197, 17] → [10, 30, 189, 219] expected [16, 17, 197, 219]
  FAIL: [30, 75, 7, 249] → [13, 35, 73, 240] expected [7, 30, 75, 249]
  FAIL: [185, 34, 187, 212] → [37, 180, 194, 210] expected [34, 185, 187, 212]
  FAIL: [162, 218, 169, 254] → [157, 175, 219, 257] expected [162, 169, 218, 254]
  FAIL: [98, 63, 21, 246] → [25, 67, 99, 239] expected [21, 63, 98, 246]
  FAIL: [68, 246, 21, 16] → [5, 21, 72, 254] expected [16, 21, 68, 246]
  FAIL: [154, 163, 218, 249] → [148, 169, 221, 249] expected [154, 163, 218, 249]
  FAIL: [247, 43, 102, 37] → [32, 49, 104, 245] expected [37, 43, 102, 247]
  FAIL: [250, 234, 230, 51] → [57, 224, 244, 244] expected [51, 230, 234, 250]
  FAIL: [197, 103, 168, 170] → [103, 166, 176, 196] expected [103, 168, 170, 197]
  FAIL: [139, 227, 20, 25] → [8, 29, 138, 237] expected [20, 25, 139, 227]
  FAIL: [22, 73, 61, 249] → [24, 60, 80, 243] expected [22, 61, 73, 249]
  FAIL: [44, 233, 160, 247] → [50, 165, 227, 246] expected [44, 160, 233, 247]
  FAIL: [128, 98, 53, 48] → [42, 55, 99, 130] expected [48, 53, 98, 128]
  FAIL: [159, 134, 248, 32] → [35, 128, 164, 247] expected [32, 134, 159, 248]

  Exact: 0/100
  Within ±5 and ordered: 60/100

============================================================
TEST 2: Close values (spacing 1-5)
============================================================

  ✗ [29, 35, 20, 25] → [19, 24, 31, 30] expected [20, 25, 29, 35]
  ✓ [53, 59, 49, 45] → [44, 48, 54, 57] expected [45, 49, 53, 59]
  ✓ [82, 71, 77, 70] → [67, 75, 77, 78] expected [70, 71, 77, 82]
  ✓ [98, 101, 95, 108] → [93, 100, 103, 103] expected [95, 98, 101, 108]
  ✓ [120, 123, 126, 134] → [116, 125, 129, 132] expected [120, 123, 126, 134]
  ✓ [149, 145, 160, 153] → [143, 150, 152, 162] expected [145, 149, 153, 160]
  ✓ [170, 178, 184, 175] → [166, 178, 182, 183] expected [170, 175, 178, 184]
  ✗ [195, 204, 197, 208] → [190, 207, 204, 205] expected [195, 197, 204, 208]
  ✗ [231, 229, 220, 225] → [222, 234, 226, 228] expected [220, 225, 229, 231]

  Close accuracy: 6/9

============================================================
TEST 3: Generalization (NEVER TRAINED)
============================================================

  Width 5:
    FAIL: [17, 86, 150, 152, 190] → [18, 80, 147, 158, 199] expected [17, 86, 150, 152, 190]
    FAIL: [154, 21, 1, 237, 160] → [-1, 72, 73, 181, 214] expected [1, 21, 154, 160, 237]
    FAIL: [131, 22, 185, 23, 231] → [3, 37, 87, 168, 270] expected [22, 23, 131, 185, 231]
    FAIL: [205, 19, 64, 207, 85] → [35, 92, 91, 146, 213] expected [19, 64, 85, 205, 207]
    FAIL: [79, 77, 138, 134, 242] → [48, 70, 138, 162, 246] expected [77, 79, 134, 138, 242]
    FAIL: [125, 178, 79, 99, 144] → [71, 86, 98, 150, 201] expected [79, 99, 125, 144, 178]
    FAIL: [254, 144, 200, 45, 18] → [82, 152, 149, 142, 171] expected [18, 45, 144, 200, 254]
    FAIL: [7, 177, 191, 45, 118] → [-1, 40, 86, 174, 249] expected [7, 45, 118, 177, 191]
    FAIL: [62, 250, 44, 248, 103] → [67, 119, 69, 205, 263] expected [44, 62, 103, 248, 250]
    FAIL: [150, 126, 181, 101, 28] → [99, 94, 112, 117, 180] expected [28, 101, 126, 150, 181]
    FAIL: [162, 47, 161, 34, 118] → [42, 56, 114, 127, 164] expected [34, 47, 118, 161, 162]
    FAIL: [222, 138, 41, 232, 96] → [29, 136, 191, 166, 229] expected [41, 96, 138, 222, 232]
    FAIL: [227, 74, 109, 200, 91] → [92, 124, 110, 162, 221] expected [74, 91, 109, 200, 227]
    FAIL: [103, 60, 162, 89, 19] → [38, 43, 72, 111, 169] expected [19, 60, 89, 103, 162]
    FAIL: [1, 83, 21, 105, 174] → [-18, 37, 73, 110, 143] expected [1, 21, 83, 105, 174]
    FAIL: [211, 47, 115, 237, 54] → [73, 121, 117, 141, 232] expected [47, 54, 115, 211, 237]
    FAIL: [143, 49, 204, 108, 9] → [42, 60, 103, 138, 190] expected [9, 49, 108, 143, 204]
    FAIL: [56, 184, 121, 183, 57] → [23, 91, 168, 163, 187] expected [56, 57, 121, 183, 184]
    FAIL: [168, 61, 1, 63, 221] → [19, 16, 35, 112, 253] expected [1, 61, 63, 168, 221]
    FAIL: [182, 90, 13, 32, 223] → [24, 16, 36, 127, 267] expected [13, 32, 90, 182, 223]
    Accuracy (±5 and ordered): 0/20

  Width 6:
    FAIL: [17, 34, 66, 97, 41, 29] → [-9, -6, 7, 18, 84, 133] expected [17, 29, 34, 41, 66, 97]
    FAIL: [55, 192, 142, 60, 30, 71] → [13, 69, 72, 90, 99, 174] expected [30, 55, 60, 71, 142, 192]
    FAIL: [14, 55, 181, 189, 122, 245] → [-4, 62, 103, 191, 231, 252] expected [14, 55, 122, 181, 189, 245]
    FAIL: [254, 124, 165, 213, 52, 30] → [125, 134, 134, 99, 148, 239] expected [30, 52, 124, 165, 213, 254]
    FAIL: [192, 51, 51, 80, 182, 245] → [46, 18, 81, 95, 220, 273] expected [51, 51, 80, 182, 192, 245]
    FAIL: [38, 58, 252, 29, 231, 174] → [19, 54, 90, 121, 207, 281] expected [29, 38, 58, 174, 231, 252]
    FAIL: [211, 207, 241, 112, 51, 153] → [124, 127, 155, 140, 198, 246] expected [51, 112, 153, 207, 211, 241]
    FAIL: [248, 66, 20, 53, 170, 170] → [40, 16, 68, 95, 201, 214] expected [20, 53, 66, 170, 170, 248]
    FAIL: [237, 144, 150, 241, 170, 145] → [123, 163, 175, 190, 225, 267] expected [144, 145, 150, 170, 237, 241]
    FAIL: [174, 118, 236, 179, 85, 44] → [81, 92, 141, 169, 193, 223] expected [44, 85, 118, 174, 179, 236]
    FAIL: [230, 4, 28, 160, 252, 58] → [50, 44, 80, 88, 191, 238] expected [4, 28, 58, 160, 230, 252]
    FAIL: [253, 19, 159, 113, 137, 144] → [35, 124, 123, 119, 165, 255] expected [19, 113, 137, 144, 159, 253]
    FAIL: [159, 24, 113, 46, 248, 237] → [10, 36, 64, 133, 221, 295] expected [24, 46, 113, 159, 237, 248]
    FAIL: [69, 16, 38, 122, 124, 44] → [-10, 13, 32, 27, 124, 160] expected [16, 38, 44, 69, 122, 124]
    FAIL: [206, 169, 213, 100, 237, 162] → [83, 163, 176, 227, 219, 266] expected [100, 162, 169, 206, 213, 237]
    FAIL: [85, 232, 208, 241, 103, 99] → [79, 168, 166, 174, 206, 248] expected [85, 99, 103, 208, 232, 241]
    FAIL: [139, 229, 84, 245, 195, 10] → [72, 129, 163, 170, 219, 188] expected [10, 84, 139, 195, 229, 245]
    FAIL: [121, 103, 199, 96, 213, 59] → [75, 69, 107, 136, 204, 216] expected [59, 96, 103, 121, 199, 213]
    FAIL: [153, 99, 56, 124, 76, 92] → [55, 92, 90, 85, 106, 146] expected [56, 76, 92, 99, 124, 153]
    FAIL: [94, 135, 150, 70, 169, 192] → [16, 94, 135, 173, 182, 204] expected [70, 94, 135, 150, 169, 192]
    Accuracy (±5 and ordered): 0/20

so it can't generalize, but also 60% accuracy on 4 wide shows that it does work in theory, it's not randomly guessing but also it can't fine tune. it can not get the exact number right because tanh applied over 100 steps looses a lot of accuracy

Ok so I did a lot more random experiments, I didn't exactly record the exact failures but here is the summary 
 - gated residuals - they preserve values a lot better than tanh
 - They diverge at about 400k itterations
 - Adam doesn't really help prevent the diverge
 - Adam with weight decay does work a lot better as weights can't really diverge 
 - That worked somehow

Using: cuda
============================================================
TEST 1: 100 random width-7 cases
============================================================
  CLOSE: [0, 125, 0, 0, 16, 0, 0] → [0, 0, 0, 0, 0, 16, 126] expected [0, 0, 0, 0, 0, 16, 125]
  CLOSE: [0, 112, 142, 0, 174, 79, 0] → [0, 0, 0, 78, 113, 143, 176] expected [0, 0, 0, 79, 112, 142, 174]
  CLOSE: [0, 49, 176, 22, 63, 40, 185] → [0, 23, 40, 49, 63, 176, 186] expected [0, 22, 40, 49, 63, 176, 185]
  CLOSE: [0, 0, 40, 0, 142, 186, 0] → [0, 0, 0, 0, 39, 143, 187] expected [0, 0, 0, 0, 40, 142, 186]
  CLOSE: [107, 36, 0, 0, 138, 0, 0] → [0, 0, 0, 0, 37, 108, 139] expected [0, 0, 0, 0, 36, 107, 138]
  CLOSE: [0, 137, 0, 108, 202, 73, 71] → [0, 0, 71, 75, 108, 137, 203] expected [0, 0, 71, 73, 108, 137, 202]
  CLOSE: [0, 204, 112, 0, 0, 0, 0] → [0, 0, 0, 0, 0, 109, 207] expected [0, 0, 0, 0, 0, 112, 204]
  CLOSE: [0, 239, 5, 0, 57, 222, 0] → [0, 0, 0, 5, 57, 224, 240] expected [0, 0, 0, 5, 57, 222, 239]
  CLOSE: [0, 0, 101, 0, 0, 250, 0] → [0, 0, 0, 0, 0, 102, 251] expected [0, 0, 0, 0, 0, 101, 250]
  CLOSE: [157, 0, 0, 0, 0, 0, 0] → [0, 0, 0, 0, 0, 0, 158] expected [0, 0, 0, 0, 0, 0, 157]
  CLOSE: [108, 0, 191, 231, 0, 0, 10] → [0, 0, 0, 9, 111, 194, 232] expected [0, 0, 0, 10, 108, 191, 231]
  CLOSE: [0, 0, 0, 0, 36, 0, 109] → [0, 0, 0, 0, 0, 35, 110] expected [0, 0, 0, 0, 0, 36, 109]
  CLOSE: [0, 0, 97, 0, 181, 210, 27] → [0, 0, 0, 26, 98, 182, 211] expected [0, 0, 0, 27, 97, 181, 210]
  CLOSE: [0, 173, 0, 0, 71, 93, 236] → [0, 0, 0, 72, 93, 174, 236] expected [0, 0, 0, 71, 93, 173, 236]
  CLOSE: [0, 50, 0, 0, 0, 246, 0] → [0, 0, 0, 0, 0, 50, 247] expected [0, 0, 0, 0, 0, 50, 246]
  CLOSE: [0, 1, 135, 146, 249, 0, 111] → [0, 0, 0, 111, 136, 147, 250] expected [0, 0, 1, 111, 135, 146, 249]
  CLOSE: [0, 0, 161, 104, 122, 202, 0] → [0, 0, 0, 105, 123, 159, 204] expected [0, 0, 0, 104, 122, 161, 202]
  CLOSE: [161, 0, 51, 0, 67, 35, 0] → [0, 0, 0, 35, 51, 67, 162] expected [0, 0, 0, 35, 51, 67, 161]
  CLOSE: [80, 154, 0, 0, 59, 0, 144] → [0, 0, 0, 59, 82, 145, 154] expected [0, 0, 0, 59, 80, 144, 154]
  CLOSE: [0, 0, 128, 0, 141, 0, 66] → [0, 0, 0, 0, 66, 129, 142] expected [0, 0, 0, 0, 66, 128, 141]
  CLOSE: [82, 218, 0, 0, 0, 0, 0] → [0, 0, 0, 0, 0, 84, 219] expected [0, 0, 0, 0, 0, 82, 218]
  CLOSE: [0, 210, 0, 0, 19, 113, 0] → [0, 0, 0, 0, 20, 113, 210] expected [0, 0, 0, 0, 19, 113, 210]
  CLOSE: [156, 0, 0, 168, 35, 179, 169] → [0, 0, 36, 157, 168, 171, 178] expected [0, 0, 35, 156, 168, 169, 179]
  CLOSE: [0, 91, 19, 0, 160, 59, 97] → [0, 0, 19, 60, 91, 97, 160] expected [0, 0, 19, 59, 91, 97, 160]
  CLOSE: [22, 0, 0, 35, 160, 0, 209] → [0, 0, 0, 21, 35, 160, 211] expected [0, 0, 0, 22, 35, 160, 209]
  CLOSE: [206, 65, 0, 89, 207, 0, 107] → [0, 0, 64, 89, 108, 207, 209] expected [0, 0, 65, 89, 107, 206, 207]
  CLOSE: [164, 226, 109, 86, 0, 47, 0] → [0, 0, 46, 87, 107, 169, 227] expected [0, 0, 47, 86, 109, 164, 226]
  CLOSE: [0, 0, 0, 37, 212, 0, 204] → [0, 0, 0, 0, 38, 206, 212] expected [0, 0, 0, 0, 37, 204, 212]
  CLOSE: [0, 0, 112, 0, 0, 0, 0] → [0, 0, 0, 0, 0, 0, 113] expected [0, 0, 0, 0, 0, 0, 112]
  CLOSE: [226, 228, 0, 132, 0, 122, 225] → [0, 0, 122, 132, 226, 227, 229] expected [0, 0, 122, 132, 225, 226, 228]
  CLOSE: [0, 0, 163, 0, 0, 78, 0] → [0, 0, 0, 0, 0, 78, 164] expected [0, 0, 0, 0, 0, 78, 163]
  CLOSE: [208, 238, 31, 0, 10, 244, 0] → [0, 0, 14, 30, 210, 239, 245] expected [0, 0, 10, 31, 208, 238, 244]
  CLOSE: [199, 112, 112, 223, 14, 172, 84] → [13, 85, 109, 115, 174, 200, 222] expected [14, 84, 112, 112, 172, 199, 223]
  CLOSE: [65, 0, 0, 69, 93, 0, 167] → [0, 0, 0, 65, 69, 93, 168] expected [0, 0, 0, 65, 69, 93, 167]
  CLOSE: [0, 172, 142, 129, 0, 0, 114] → [0, 0, 0, 116, 131, 139, 174] expected [0, 0, 0, 114, 129, 142, 172]
  CLOSE: [0, 0, 0, 0, 0, 0, 131] → [0, 0, 0, 0, 0, 0, 132] expected [0, 0, 0, 0, 0, 0, 131]
  CLOSE: [85, 0, 55, 0, 203, 0, 0] → [0, 0, 0, 0, 53, 87, 205] expected [0, 0, 0, 0, 55, 85, 203]
  CLOSE: [61, 0, 189, 0, 0, 54, 185] → [0, 0, 0, 55, 61, 186, 189] expected [0, 0, 0, 54, 61, 185, 189]
  CLOSE: [78, 90, 247, 223, 165, 0, 230] → [0, 77, 90, 167, 225, 232, 245] expected [0, 78, 90, 165, 223, 230, 247]
  CLOSE: [0, 172, 0, 93, 108, 132, 143] → [0, 0, 94, 109, 133, 144, 170] expected [0, 0, 93, 108, 132, 143, 172]
  CLOSE: [5, 0, 0, 123, 251, 8, 0] → [0, 0, 0, 4, 8, 125, 252] expected [0, 0, 0, 5, 8, 123, 251]
  CLOSE: [0, 0, 242, 217, 180, 138, 128] → [0, 0, 130, 138, 179, 216, 243] expected [0, 0, 128, 138, 180, 217, 242]
  CLOSE: [0, 0, 0, 0, 141, 51, 0] → [0, 0, 0, 0, 0, 51, 142] expected [0, 0, 0, 0, 0, 51, 141]
  CLOSE: [240, 225, 94, 0, 58, 0, 37] → [0, 0, 37, 60, 89, 228, 243] expected [0, 0, 37, 58, 94, 225, 240]
  CLOSE: [240, 16, 0, 232, 0, 101, 58] → [0, 0, 15, 59, 102, 234, 240] expected [0, 0, 16, 58, 101, 232, 240]
  CLOSE: [0, 72, 0, 0, 224, 0, 206] → [0, 0, 0, 0, 72, 207, 225] expected [0, 0, 0, 0, 72, 206, 224]
  CLOSE: [252, 41, 0, 128, 0, 0, 20] → [0, 0, 0, 20, 43, 128, 253] expected [0, 0, 0, 20, 41, 128, 252]
  CLOSE: [0, 142, 0, 46, 178, 170, 53] → [0, 0, 47, 54, 141, 171, 178] expected [0, 0, 46, 53, 142, 170, 178]
  CLOSE: [0, 253, 205, 0, 0, 165, 0] → [0, 0, 0, 0, 165, 205, 254] expected [0, 0, 0, 0, 165, 205, 253]
  CLOSE: [0, 27, 0, 226, 0, 67, 224] → [0, 0, 0, 26, 68, 225, 226] expected [0, 0, 0, 27, 67, 224, 226]
  FAIL: [149, 212, 241, 0, 0, 0, 0] → [0, 0, 0, -1, 152, 214, 242] expected [0, 0, 0, 0, 149, 212, 241]
  CLOSE: [174, 1, 152, 76, 247, 170, 233] → [0, 76, 154, 172, 172, 233, 246] expected [1, 76, 152, 170, 174, 233, 247]
  CLOSE: [96, 0, 0, 0, 195, 77, 18] → [0, 0, 0, 19, 77, 96, 196] expected [0, 0, 0, 18, 77, 96, 195]
  CLOSE: [0, 0, 0, 0, 79, 0, 173] → [0, 0, 0, 0, 0, 79, 174] expected [0, 0, 0, 0, 0, 79, 173]
  CLOSE: [41, 194, 249, 0, 0, 0, 50] → [0, 0, 0, 43, 50, 197, 251] expected [0, 0, 0, 41, 50, 194, 249]
  CLOSE: [0, 0, 0, 28, 183, 220, 0] → [0, 0, 0, 0, 27, 184, 221] expected [0, 0, 0, 0, 28, 183, 220]
  CLOSE: [92, 0, 0, 0, 0, 130, 130] → [0, 0, 0, 0, 92, 129, 132] expected [0, 0, 0, 0, 92, 130, 130]
  CLOSE: [0, 80, 0, 153, 128, 154, 0] → [0, 0, 0, 80, 127, 153, 156] expected [0, 0, 0, 80, 128, 153, 154]
  CLOSE: [54, 0, 151, 11, 140, 0, 146] → [0, 0, 11, 54, 141, 147, 151] expected [0, 0, 11, 54, 140, 146, 151]
  CLOSE: [0, 0, 70, 0, 225, 0, 0] → [0, 0, 0, 0, 0, 71, 227] expected [0, 0, 0, 0, 0, 70, 225]
  CLOSE: [167, 89, 0, 139, 0, 151, 58] → [0, 0, 57, 90, 141, 153, 167] expected [0, 0, 58, 89, 139, 151, 167]
  CLOSE: [38, 0, 187, 0, 0, 0, 134] → [0, 0, 0, 0, 38, 135, 188] expected [0, 0, 0, 0, 38, 134, 187]
  CLOSE: [190, 0, 12, 113, 0, 209, 0] → [0, 0, 0, 13, 115, 189, 211] expected [0, 0, 0, 12, 113, 190, 209]
  CLOSE: [0, 252, 0, 0, 232, 214, 0] → [0, 0, 0, 0, 216, 233, 250] expected [0, 0, 0, 0, 214, 232, 252]
  CLOSE: [0, 143, 0, 0, 120, 50, 183] → [0, 0, 0, 51, 121, 143, 184] expected [0, 0, 0, 50, 120, 143, 183]
  CLOSE: [0, 97, 0, 0, 0, 124, 0] → [0, 0, 0, 0, 0, 97, 125] expected [0, 0, 0, 0, 0, 97, 124]
  CLOSE: [0, 121, 8, 0, 0, 58, 0] → [0, 0, 0, 0, 7, 58, 122] expected [0, 0, 0, 0, 8, 58, 121]
  CLOSE: [185, 0, 0, 234, 0, 205, 55] → [0, 0, 0, 56, 185, 207, 234] expected [0, 0, 0, 55, 185, 205, 234]
  CLOSE: [227, 0, 75, 0, 166, 150, 220] → [0, 0, 79, 150, 167, 221, 225] expected [0, 0, 75, 150, 166, 220, 227]
  CLOSE: [0, 0, 116, 173, 204, 48, 218] → [0, 0, 49, 116, 173, 204, 219] expected [0, 0, 48, 116, 173, 204, 218]
  CLOSE: [130, 78, 34, 0, 0, 0, 0] → [0, 0, 0, 0, 33, 77, 132] expected [0, 0, 0, 0, 34, 78, 130]
  CLOSE: [62, 181, 26, 159, 53, 0, 114] → [0, 27, 54, 62, 114, 161, 180] expected [0, 26, 53, 62, 114, 159, 181]
  CLOSE: [0, 58, 115, 13, 14, 0, 173] → [0, 0, 10, 17, 59, 115, 174] expected [0, 0, 13, 14, 58, 115, 173]
  CLOSE: [3, 0, 35, 0, 0, 214, 174] → [0, 0, 0, 2, 34, 174, 214] expected [0, 0, 0, 3, 35, 174, 214]
  CLOSE: [0, 166, 0, 0, 0, 226, 248] → [0, 0, 0, 0, 166, 228, 247] expected [0, 0, 0, 0, 166, 226, 248]
  CLOSE: [212, 110, 0, 56, 249, 23, 0] → [0, 0, 21, 57, 110, 215, 250] expected [0, 0, 23, 56, 110, 212, 249]
  CLOSE: [0, 0, 0, 148, 61, 0, 89] → [0, 0, 0, 0, 61, 90, 149] expected [0, 0, 0, 0, 61, 89, 148]
  CLOSE: [0, 0, 0, 0, 0, 0, 207] → [0, 0, 0, 0, 0, 0, 208] expected [0, 0, 0, 0, 0, 0, 207]
  CLOSE: [148, 0, 0, 0, 45, 54, 0] → [0, 0, 0, 0, 47, 53, 148] expected [0, 0, 0, 0, 45, 54, 148]
  CLOSE: [40, 158, 113, 86, 0, 0, 75] → [0, 0, 39, 76, 88, 111, 160] expected [0, 0, 40, 75, 86, 113, 158]
  CLOSE: [0, 0, 0, 0, 0, 229, 161] → [0, 0, 0, 0, 0, 161, 230] expected [0, 0, 0, 0, 0, 161, 229]
  CLOSE: [0, 0, 155, 28, 37, 236, 19] → [0, 0, 20, 28, 38, 155, 237] expected [0, 0, 19, 28, 37, 155, 236]
  CLOSE: [0, 39, 0, 21, 96, 243, 0] → [0, 0, 0, 21, 37, 97, 244] expected [0, 0, 0, 21, 39, 96, 243]
  CLOSE: [52, 43, 0, 0, 81, 190, 198] → [0, 0, 45, 51, 82, 190, 198] expected [0, 0, 43, 52, 81, 190, 198]
  CLOSE: [173, 0, 0, 0, 129, 0, 0] → [0, 0, 0, 0, 0, 129, 174] expected [0, 0, 0, 0, 0, 129, 173]
  CLOSE: [158, 66, 0, 168, 0, 185, 0] → [0, 0, 0, 67, 158, 169, 186] expected [0, 0, 0, 66, 158, 168, 185]
  CLOSE: [92, 0, 98, 0, 0, 51, 0] → [0, 0, 0, 0, 50, 94, 98] expected [0, 0, 0, 0, 51, 92, 98]
  CLOSE: [0, 0, 0, 22, 186, 0, 229] → [0, 0, 0, 0, 22, 186, 231] expected [0, 0, 0, 0, 22, 186, 229]
  CLOSE: [0, 240, 0, 236, 195, 82, 0] → [0, 0, 0, 82, 195, 239, 242] expected [0, 0, 0, 82, 195, 236, 240]
  CLOSE: [26, 190, 0, 42, 0, 75, 46] → [0, 0, 25, 43, 46, 75, 190] expected [0, 0, 26, 42, 46, 75, 190]
  CLOSE: [0, 13, 27, 191, 0, 0, 0] → [0, 0, 0, 0, 13, 28, 193] expected [0, 0, 0, 0, 13, 27, 191]
  CLOSE: [50, 74, 0, 241, 0, 2, 0] → [0, 0, 0, 0, 50, 77, 243] expected [0, 0, 0, 2, 50, 74, 241]
  CLOSE: [110, 0, 0, 0, 0, 0, 32] → [0, 0, 0, 0, 0, 32, 111] expected [0, 0, 0, 0, 0, 32, 110]

  Exact: 7/100
  Within ±5 and ordered: 99/100

============================================================
TEST 2: Width-5 (never trained)
============================================================
  Accuracy: 20/20

============================================================
TEST 3: Width-10 (never trained)
============================================================
  FAIL: [245, 0, 139, 0, 0, 0, 0, 0, 241, 0] → [0, 0, 0, 0, 0, 0, 0, 151, 241, 243] expected [0, 0, 0, 0, 0, 0, 0, 139, 241, 245]
  FAIL: [0, 243, 102, 164, 30, 0, 0, 90, 0, 95] → [0, 0, 0, 0, 24, 93, 100, 104, 167, 244] expected [0, 0, 0, 0, 30, 90, 95, 102, 164, 243]
  FAIL: [78, 191, 241, 0, 0, 39, 0, 0, 0, 59] → [0, 0, 0, 0, 0, 46, 63, 74, 199, 242] expected [0, 0, 0, 0, 0, 39, 59, 78, 191, 241]
  Accuracy: 17/20

100% accuracy on width 5, probably because width 5 is an indirect subset of width 7, which is what it's trained on
but 85% on a way bigger matrix shows that it does generalize, but the info might not have had enough time to get close to exact value. but it still got good generalization