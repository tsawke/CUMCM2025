@REM python Q3Solver_v1.py ^
@REM   --dt 0.001 ^
@REM   --nphi 960 ^
@REM   --nz 13 ^
@REM   --backend process ^
@REM   --workers 8 ^
@REM   --chunk 1200

@REM python Q3Solver_v1.py ^
@REM   --dt 0.0005 ^
@REM   --nphi 960 ^
@REM   --nz 13 ^
@REM   --backend process ^
@REM   --chunk 1200

@REM python Q3Solver_v2.py ^
@REM   --dt 0.0015 --nphi 720 --nz 11 ^
@REM   --outer_backend process ^
@REM   --backend thread --workers 1 --chunk 1200 --block 8192 ^
@REM   --heading_min_deg -2.0 --heading_max_deg 2.0 --heading_step_deg 0.5 ^
@REM   --speed_min 120 --speed_max 140 --speed_step 10 ^
@REM   --t1_min 0 --t1_max 6 --t1_step 1 ^
@REM   --gap12_min 1 --gap12_max 4 --gap12_step 1 ^
@REM   --gap23_min 1 --gap23_max 4 --gap23_step 1 ^
@REM   --fuse_min 3 --fuse_max 8 --fuse_step 1 ^
@REM   --inner_log --inner_log_every 10

@REM python Q3Solver.py ^
@REM   --dt 0.01 --nphi 60 --nz 3 ^
@REM   --outer_backend process  ^
@REM   --backend thread --workers 1 --chunk 400 --block 1024 ^
@REM   --heading_min_deg -1.0 --heading_max_deg 1.0 --heading_step_deg 1.0 ^
@REM   --speed_min 130 --speed_max 140 --speed_step 10 ^
@REM   --t1_min 0 --t1_max 2 --t1_step 1 ^
@REM   --gap12_min 1 --gap12_max 2 --gap12_step 1 ^
@REM   --gap23_min 1 --gap23_max 2 --gap23_step 1 ^
@REM   --fuse_min 3 --fuse_max 5 --fuse_step 1


@REM python Q3Solver.py ^
@REM   --dt 0.005 --nphi 240 --nz 7 ^
@REM   --outer_backend process --search_workers 8 ^
@REM   --backend thread --workers 1 --chunk 800 --block 4096 ^
@REM   --heading_min_deg -2.0 --heading_max_deg 2.0 --heading_step_deg 0.5 ^
@REM   --speed_min 120 --speed_max 140 --speed_step 10 ^
@REM   --t1_min 0 --t1_max 4 --t1_step 1 ^
@REM   --gap12_min 1 --gap12_max 3 --gap12_step 1 ^
@REM   --gap23_min 1 --gap23_max 3 --gap23_step 1 ^
@REM   --fuse_min 3 --fuse_max 6 --fuse_step 1


@REM python Q3Solver.py ^
@REM   --dt 0.01 --nphi 120 --nz 5 ^
@REM   --outer_backend process --search_workers 8 ^
@REM   --backend thread --workers 1 --chunk 600 --block 2048 ^
@REM   --heading_min_deg -1 --heading_max_deg 1 --heading_step_deg 1 ^
@REM   --speed_min 130 --speed_max 140 --speed_step 10 ^
@REM   --t1_min 0 --t1_max 3 --t1_step 1 ^
@REM   --gap12_min 1 --gap12_max 3 --gap12_step 1 ^
@REM   --gap23_min 1 --gap23_max 3 --gap23_step 1 ^
@REM   --fuse_min 3 --fuse_max 6 --fuse_step 1 ^
@REM   --stage2_enable --topk 10 --stage2_span_steps 2 ^
@REM   --inner_log --inner_log_every 10


@REM python Q3Solver.py ^
@REM   --dt 0.01 --nphi 120 --nz 5 ^
@REM   --outer_backend process --search_workers 6 ^
@REM   --backend thread --workers 1 --chunk 2000 --block 2048 ^
@REM   --heading_min_deg -1 --heading_max_deg 1 --heading_step_deg 1 ^
@REM   --speed_min 130 --speed_max 140 --speed_step 10 ^
@REM   --t1_min 0 --t1_max 3 --t1_step 1 ^
@REM   --gap12_min 1 --gap12_max 3 --gap12_step 1 ^
@REM   --gap23_min 1 --gap23_max 3 --gap23_step 1 ^
@REM   --fuse_min 3 --fuse_max 6 --fuse_step 1 ^
@REM   --stage2_enable --stage2_span_steps 1 ^
@REM   --stage2_heading_step_deg 0.2 --stage2_speed_step 5 --stage2_t_step 0.5 --stage2_fuse_step 1.0 ^
@REM   --stage2_max_cands 8000


@REM python Q3Solver.py ^
@REM   --dt 0.01 --nphi 120 --nz 5 ^
@REM   --outer_backend process --search_workers 6 ^
@REM   --backend thread --workers 1 --chunk 1500 --block 2048 ^
@REM   --heading_min_deg -1 --heading_max_deg 1 --heading_step_deg 1 ^
@REM   --speed_min 130 --speed_max 140 --speed_step 10 ^
@REM   --t1_min 0 --t1_max 2 --t1_step 1 ^
@REM   --gap12_min 1 --gap12_max 2 --gap12_step 1 ^
@REM   --gap23_min 1 --gap23_max 2 --gap23_step 1 ^
@REM   --fuse_min 3 --fuse_max 5 --fuse_step 1


@REM python Q3Solver.py ^
@REM   --dt 0.01 --nphi 120 --nz 5 ^
@REM   --outer_backend process --search_workers 6 ^
@REM   --backend thread --workers 1 --chunk 1500 --block 2048 ^
@REM   --heading_min_deg -1 --heading_max_deg 1 --heading_step_deg 1 ^
@REM   --speed_min 130 --speed_max 140 --speed_step 10 ^
@REM   --t1_min 0 --t1_max 2 --t1_step 1 ^
@REM   --gap12_min 1 --gap12_max 2 --gap12_step 1 ^
@REM   --gap23_min 1 --gap23_max 2 --gap23_step 1 ^
@REM   --fuse_min 3 --fuse_max 5 --fuse_step 1

@REM python Q3Solver.py ^
@REM   --dt 0.008 --nphi 180 --nz 6 ^
@REM   --outer_backend process ^
@REM   --backend thread --workers 1 --chunk 1800 --block 3072 ^
@REM   --heading_min_deg -2 --heading_max_deg 2 --heading_step_deg 0.5 ^
@REM   --speed_min 120 --speed_max 140 --speed_step 10 ^
@REM   --t1_min 0 --t1_max 4 --t1_step 1 ^
@REM   --gap12_min 1 --gap12_max 3 --gap12_step 1 ^
@REM   --gap23_min 1 --gap23_max 3 --gap23_step 1 ^
@REM   --fuse_min 3 --fuse_max 6 --fuse_step 1


@REM python Q3Solver.py ^
@REM   --dt 0.005 --nphi 360 --nz 7 ^
@REM   --outer_backend process ^
@REM   --backend thread --workers 1 --chunk 2200 --block 4096 ^
@REM   --heading_min_deg -2 --heading_max_deg 2 --heading_step_deg 0.5 ^
@REM   --speed_min 120 --speed_max 140 --speed_step 10 ^
@REM   --t1_min 0 --t1_max 6 --t1_step 1 ^
@REM   --gap12_min 1 --gap12_max 4 --gap12_step 1 ^
@REM   --gap23_min 1 --gap23_max 4 --gap23_step 1 ^
@REM   --fuse_min 3 --fuse_max 8 --fuse_step 1 ^
@REM   --stage2_enable --stage2_span_steps 1 ^
@REM   --stage2_heading_step_deg 0.2 --stage2_speed_step 5 --stage2_t_step 0.5 --stage2_fuse_step 0.5 ^
@REM   --stage2_batch_size 2000 --stage2_max_cands 8000

python Q3Solver_v3.py --heading_min_deg -0.5 --heading_max_deg 0.5 --heading_step_deg 0.2 --speed_min 130 --speed_max 140 --speed_step 5 --t1_min 0 --t1_max 2 --t1_step 1 --gap12_min 3 --gap12_max 5 --gap12_step 1 --gap23_min 3 --gap23_max 5 --gap23_step 1 --fuse_min 3 --fuse_max 8 --fuse_step 1 --no_cd --workers 28