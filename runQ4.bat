python Q4Solver_origin.py ^
  --backend process --workers 28 --intra-threads 1 ^
  --dt 0.0030 --dt-coarse 0.010 ^
  --nphi 240 --nz 3 ^
  --fy1-te 7,9,0.08 --fy2-te 14,18,0.08 --fy3-te 26,32,0.08 ^
  --speed-grid 140,120,100 ^
  --lat-scales 0.0,0.015,-0.015 ^
  --topk 6 --min-gap 0.40 --combo-topm 8 ^
  --sa-iters 20 --sa-batch 28 --sa-T0 1.0 --sa-alpha 0.92 ^
  --sigma-Te 0.45 --sigma-v 8.0 --local-iters 6
