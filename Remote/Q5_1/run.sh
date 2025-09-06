python3 Q5Solver.py --use_numba --numba_threads $(nproc) \
  --dt 0.01 --nphi 720 --nz 11 \
  --sa_iters 1200 --sa_batch 16 --sa_restarts 2 \
  --progress_every 15