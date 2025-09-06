python3 Q3Solver.py \
  --dt 0.008 --nphi 180 --nz 6 \
  --outer_backend process --search_workers $(nproc) \
  --backend thread --workers 1 --chunk 1800 --block 3072 \
  --heading_min_deg -2 --heading_max_deg 2 --heading_step_deg 0.5 \
  --speed_min 120 --speed_max 140 --speed_step 10 \
  --t1_min 0 --t1_max 4 --t1_step 1 \
  --gap12_min 1 --gap12_max 3 --gap12_step 1 \
  --gap23_min 1 --gap23_max 3 --gap23_step 1 \
  --fuse_min 3 --fuse_max 6 --fuse_step 1
