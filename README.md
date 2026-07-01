evo_ape kitti 05.txt trajectory.txt -va --align --plot --plot_mode xyz
--------------------------------------------------------------------------------
Loaded 2761 poses from: 05.txt
Loaded 2761 poses from: trajectory.txt
--------------------------------------------------------------------------------
Aligning using Umeyama's method...
Rotation of alignment:
[[ 0.99982926 -0.0184329   0.00129506]
 [ 0.01837171  0.99913363  0.03734265]
 [-0.00198227 -0.03731248  0.99930168]]
Translation of alignment:
[-1.31612579 -5.69974821 -2.47996168]
Scale correction: 1.0
--------------------------------------------------------------------------------
Compared 2761 absolute pose pairs.
Calculating APE for translation part pose relation...
--------------------------------------------------------------------------------
APE w.r.t. translation part (m)
(with SE(3) Umeyama alignment)

       max	6.597513
      mean	2.848607
    median	2.800758
       min	0.361079
      rmse	3.089628
       sse	26355.958256
       std	1.196345

--------------------------------------------------------------------------------
Plotting results... 

evo_rpe kitti 05.txt trajectory.txt -va --align --plot --plot_mode xyz 
--------------------------------------------------------------------------------
Loaded 2761 poses from: 05.txt
Loaded 2761 poses from: trajectory.txt
--------------------------------------------------------------------------------
Aligning using Umeyama's method...
Rotation of alignment:
[[ 0.99982926 -0.0184329   0.00129506]
 [ 0.01837171  0.99913363  0.03734265]
 [-0.00198227 -0.03731248  0.99930168]]
Translation of alignment:
[-1.31612579 -5.69974821 -2.47996168]
Scale correction: 1.0
--------------------------------------------------------------------------------
Found 2760 pairs with delta 1 (frames) among 2761 poses using consecutive pairs.
Compared 2760 relative pose pairs, delta = 1 (frames) with consecutive pairs.
Calculating RPE for translation part pose relation...
--------------------------------------------------------------------------------
RPE w.r.t. translation part (m)
for delta = 1 (frames) using consecutive pairs
(with SE(3) Umeyama alignment)

       max	0.895526
      mean	0.023763
    median	0.016278
       min	0.000335
      rmse	0.039072
       sse	4.213402
       std	0.031015

--------------------------------------------------------------------------------
Plotting results... 

evo_ape kitti 07.txt trajectory.txt -va --align --plot --plot_mode xyz 
--------------------------------------------------------------------------------
Loaded 1101 poses from: 07.txt
Loaded 1101 poses from: trajectory.txt
--------------------------------------------------------------------------------
Aligning using Umeyama's method...
Rotation of alignment:
[[ 0.99997021  0.00557198  0.00534159]
 [-0.00555411  0.99997895 -0.00335369]
 [-0.00536017  0.00332392  0.99998011]]
Translation of alignment:
[ 1.45923226  0.91550129 -1.34862902]
Scale correction: 1.0
--------------------------------------------------------------------------------
Compared 1101 absolute pose pairs.
Calculating APE for translation part pose relation...
--------------------------------------------------------------------------------
APE w.r.t. translation part (m)
(with SE(3) Umeyama alignment)

       max	2.543365
      mean	1.738563
    median	1.815282
       min	0.484722
      rmse	1.812976
       sse	3618.855435
       std	0.514082

--------------------------------------------------------------------------------
Plotting results... 

evo_rpe kitti 07.txt trajectory.txt -va --align --plot --plot_mode xyz 
--------------------------------------------------------------------------------
Loaded 1101 poses from: 07.txt
Loaded 1101 poses from: trajectory.txt
--------------------------------------------------------------------------------
Aligning using Umeyama's method...
Rotation of alignment:
[[ 0.99997021  0.00557198  0.00534159]
 [-0.00555411  0.99997895 -0.00335369]
 [-0.00536017  0.00332392  0.99998011]]
Translation of alignment:
[ 1.45923226  0.91550129 -1.34862902]
Scale correction: 1.0
--------------------------------------------------------------------------------
Found 1100 pairs with delta 1 (frames) among 1101 poses using consecutive pairs.
Compared 1100 relative pose pairs, delta = 1 (frames) with consecutive pairs.
Calculating RPE for translation part pose relation...
--------------------------------------------------------------------------------
RPE w.r.t. translation part (m)
for delta = 1 (frames) using consecutive pairs
(with SE(3) Umeyama alignment)

       max	1.061646
      mean	0.023341
    median	0.013115
       min	0.000472
      rmse	0.051507
       sse	2.918247
       std	0.045915

--------------------------------------------------------------------------------
Plotting results... 
