python3 Q5ExSolver.py \
  --use_numba --fp32 \
  --dt 0.01 --nphi 720 --nz 11 \
  --sa_iters 800 --sa_batch 16 --sa_restarts 2 \
  --batch_parallel --outer_workers 4 --numba_threads 4 \
  --progress_every 20
