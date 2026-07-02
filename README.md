evo_ape kitti 05.txt trajectory.txt -va --plot --plot_mode xyz 
--------------------------------------------------------------------------------
Loaded 2761 poses from: 05.txt
Loaded 2761 poses from: trajectory.txt
--------------------------------------------------------------------------------
Aligning using Umeyama's method...
Rotation of alignment:
[[ 0.99988085 -0.01394306  0.00662341]
 [ 0.01384098  0.99978844  0.01521553]
 [-0.00683416 -0.01512204  0.9998623 ]]
Translation of alignment:
[-1.32212646 -4.11812313 -1.28259846]
Scale correction: 1.0
--------------------------------------------------------------------------------
Compared 2761 absolute pose pairs.
Calculating APE for translation part pose relation...
--------------------------------------------------------------------------------
APE w.r.t. translation part (m)
(with SE(3) Umeyama alignment)

       max	4.511321
      mean	2.615032
    median	2.688763
       min	1.154218
      rmse	2.732802
       sse	20619.722600
       std	0.793611

--------------------------------------------------------------------------------
Plotting results... 

evo_rpe kitti 05.txt trajectory.txt -va --plot --plot_mode xyz 
--------------------------------------------------------------------------------
Loaded 2761 poses from: 05.txt
Loaded 2761 poses from: trajectory.txt
--------------------------------------------------------------------------------
Aligning using Umeyama's method...
Rotation of alignment:
[[ 0.99988085 -0.01394306  0.00662341]
 [ 0.01384098  0.99978844  0.01521553]
 [-0.00683416 -0.01512204  0.9998623 ]]
Translation of alignment:
[-1.32212646 -4.11812313 -1.28259846]
Scale correction: 1.0
--------------------------------------------------------------------------------
Found 2760 pairs with delta 1 (frames) among 2761 poses using consecutive pairs.
Compared 2760 relative pose pairs, delta = 1 (frames) with consecutive pairs.
Calculating RPE for translation part pose relation...
--------------------------------------------------------------------------------
RPE w.r.t. translation part (m)
for delta = 1 (frames) using consecutive pairs
(with SE(3) Umeyama alignment)

       max	0.578333
      mean	0.019049
    median	0.014317
       min	0.000458
      rmse	0.027990
       sse	2.162276
       std	0.020508

--------------------------------------------------------------------------------
Plotting results... 

evo_ape kitti 07.txt trajectory.txt -va --plot --plot_mode xyz 
--------------------------------------------------------------------------------
Loaded 1101 poses from: 07.txt
Loaded 1101 poses from: trajectory.txt
--------------------------------------------------------------------------------
Aligning using Umeyama's method...
Rotation of alignment:
[[ 9.99922885e-01 -9.50998420e-04  1.23822534e-02]
 [ 9.99728528e-04  9.99991778e-01 -3.92988500e-03]
 [-1.23784142e-02  3.94196083e-03  9.99915614e-01]]
Translation of alignment:
[ 1.22596583 -0.19785445 -1.77335775]
Scale correction: 1.0
--------------------------------------------------------------------------------
Compared 1101 absolute pose pairs.
Calculating APE for translation part pose relation...
--------------------------------------------------------------------------------
APE w.r.t. translation part (m)
(with SE(3) Umeyama alignment)

       max	2.497478
      mean	0.981290
    median	0.848789
       min	0.086114
      rmse	1.145281
       sse	1444.147264
       std	0.590541

--------------------------------------------------------------------------------
Plotting results... 

evo_rpe kitti 07.txt trajectory.txt -va --plot --plot_mode xyz 
--------------------------------------------------------------------------------
Loaded 1101 poses from: 07.txt
Loaded 1101 poses from: trajectory.txt
--------------------------------------------------------------------------------
Aligning using Umeyama's method...
Rotation of alignment:
[[ 9.99922885e-01 -9.50998420e-04  1.23822534e-02]
 [ 9.99728528e-04  9.99991778e-01 -3.92988500e-03]
 [-1.23784142e-02  3.94196083e-03  9.99915614e-01]]
Translation of alignment:
[ 1.22596583 -0.19785445 -1.77335775]
Scale correction: 1.0
--------------------------------------------------------------------------------
Found 1100 pairs with delta 1 (frames) among 1101 poses using consecutive pairs.
Compared 1100 relative pose pairs, delta = 1 (frames) with consecutive pairs.
Calculating RPE for translation part pose relation...
--------------------------------------------------------------------------------
RPE w.r.t. translation part (m)
for delta = 1 (frames) using consecutive pairs
(with SE(3) Umeyama alignment)

       max	0.589531
      mean	0.017590
    median	0.011333
       min	0.000198
      rmse	0.037407
       sse	1.539245
       std	0.033014

--------------------------------------------------------------------------------
Plotting results... 
