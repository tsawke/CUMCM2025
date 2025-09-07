python Q2Solver.py ^
  --algo sa ^
  --pop 48 --iter 40 --topk 10 --workers auto ^
  --dt-coarse 0.003 --nphi-coarse 360 --nz-coarse 9 ^
  --dt-final  0.001 --nphi-final  720 --nz-final  11 ^
  --sa-iters 2000 --sa-batch 8 --sa-chains 2 ^
  --probe 192