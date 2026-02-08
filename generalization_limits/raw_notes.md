# Raw Notes from before during and after experiments
## read documentation.md if you want to read a cleaned up version
Pre coding
 - NCA's can generalize, but how much?
 - can NCA's generalize 3 digit addition, when only taught on 2 digit addition? (Main question)
 - can carry prop work on training data never before seen?

Architechure 
 - 16 channels
 - 2 width, 11 length
 - train on 100k itterations of random numbers added from 0 - 99
 - testing if the weights are accurate for 1000 random 3 digit addition problems

Post Coding
 - Training data (0-99): 500/500
Numbers added were 253 + 743
Numbers added were 257 + 511
Numbers added were 325 + 955
Numbers added were 933 + 987
Numbers added were 651 + 374
Numbers added were 971 + 309
Numbers added were 180 + 846
Numbers added were 890 + 903
Numbers added were 143 + 884
Numbers added were 757 + 270
Numbers added were 763 + 228
Numbers added were 891 + 901
Numbers added were 645 + 763
Numbers added were 587 + 442
Numbers added were 737 + 287
Numbers added were 863 + 931
Numbers added were 145 + 495
Numbers added were 468 + 552
Numbers added were 687 + 851
Numbers added were 573 + 457
Numbers added were 222 + 807
Numbers added were 815 + 723
Numbers added were 193 + 191
Numbers added were 462 + 560
Numbers added were 895 + 897
Numbers added were 874 + 150
Numbers added were 589 + 947
Numbers added were 929 + 607
Numbers added were 194 + 831
Numbers added were 774 + 255
Numbers added were 234 + 743
Numbers added were 836 + 189
Numbers added were 738 + 239
Numbers added were 457 + 564
Numbers added were 521 + 508
Numbers added were 305 + 207
Numbers added were 403 + 877
Numbers added were 738 + 238
Numbers added were 601 + 935
Numbers added were 678 + 349
Numbers added were 527 + 505
Numbers added were 355 + 669
Numbers added were 315 + 966
Numbers added were 687 + 851
Numbers added were 582 + 444
Numbers added were 437 + 588
Numbers added were 838 + 191
Numbers added were 422 + 859
Numbers added were 415 + 612
Numbers added were 771 + 767
Numbers added were 911 + 753
Numbers added were 879 + 913
Numbers added were 228 + 748
Numbers added were 456 + 567
Numbers added were 412 + 614
Numbers added were 234 + 790
Numbers added were 795 + 486
Numbers added were 770 + 511
Numbers added were 869 + 159
Numbers added were 236 + 752
Numbers added were 399 + 881
Numbers added were 249 + 751
Numbers added were 942 + 595
Numbers added were 572 + 964
Numbers added were 741 + 252
Numbers added were 497 + 526
Numbers added were 623 + 915
Numbers added were 807 + 221
Numbers added were 391 + 637
Numbers added were 814 + 210
Numbers added were 411 + 615
Numbers added were 193 + 830
Numbers added were 636 + 391
Numbers added were 745 + 246
Numbers added were 314 + 199
Numbers added were 790 + 236
Numbers added were 922 + 102
Numbers added were 575 + 964
Numbers added were 306 + 975
Numbers added were 254 + 737
Numbers added were 234 + 745
Numbers added were 737 + 252
Numbers added were 837 + 186
Numbers added were 349 + 678
Numbers added were 230 + 761
Numbers added were 927 + 481
Numbers added were 397 + 625
Numbers added were 765 + 259
Numbers added were 787 + 494

 - Iteration 73000 | 4+70=74 | Loss: 0.000028
Logit range: [-14.44, 10.88]
Iteration 74000 | 45+84=129 | Loss: 0.000048
Logit range: [-14.27, 11.49]
Iteration 75000 | 83+64=147 | Loss: 0.000026
Logit range: [-11.92, 10.25]
Iteration 76000 | 66+7=73 | Loss: 0.000210
Logit range: [-14.04, 11.76]
Iteration 77000 | 90+65=155 | Loss: 0.000033
Logit range: [-12.26, 10.74]
Iteration 78000 | 15+62=77 | Loss: 0.000041
Logit range: [-12.44, 11.34]
Iteration 79000 | 63+88=151 | Loss: 0.000043
Logit range: [-12.59, 12.10]
Iteration 80000 | 91+77=168 | Loss: 0.000034
Logit range: [-12.65, 10.91]
Iteration 81000 | 79+55=134 | Loss: 0.000031
Logit range: [-12.86, 10.96]
Iteration 82000 | 18+88=106 | Loss: 0.000028
Logit range: [-13.33, 10.99]
Iteration 83000 | 67+70=137 | Loss: 0.000029
Logit range: [-13.28, 11.24]
Iteration 84000 | 22+98=120 | Loss: 0.000035
Logit range: [-13.55, 12.77]
Iteration 85000 | 39+36=75 | Loss: 0.000010
Logit range: [-12.98, 8.86]
Iteration 86000 | 19+13=32 | Loss: 0.000032
Logit range: [-12.41, 11.08]
Iteration 87000 | 4+25=29 | Loss: 0.000027
Logit range: [-13.14, 13.70]
Iteration 88000 | 25+31=56 | Loss: 0.000055
Logit range: [-14.78, 10.74]
Iteration 89000 | 58+72=130 | Loss: 0.000017
Logit range: [-12.81, 12.30]
Iteration 90000 | 76+97=173 | Loss: 0.000032
Logit range: [-12.94, 9.94]
Iteration 91000 | 93+43=136 | Loss: 0.000088
Logit range: [-12.88, 10.60]
Iteration 92000 | 33+10=43 | Loss: 0.000018
Logit range: [-12.91, 10.77]
Iteration 93000 | 67+6=73 | Loss: 0.000013
Logit range: [-13.03, 14.51]
Iteration 94000 | 27+87=114 | Loss: 0.000016
Logit range: [-14.75, 13.22]
Iteration 95000 | 85+82=167 | Loss: 0.000013
Logit range: [-13.86, 11.77]
Iteration 96000 | 85+53=138 | Loss: 0.000023
Logit range: [-13.21, 11.23]
Iteration 97000 | 24+55=79 | Loss: 0.000015
Logit range: [-13.25, 12.58]
Iteration 98000 | 39+58=97 | Loss: 0.000079
Logit range: [-13.26, 8.60]
Iteration 99000 | 87+45=132 | Loss: 0.000047
