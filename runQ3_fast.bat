@echo off
echo Running Q3 Solver with time and evaluation limits...
python Q3Solver_v3.py ^
  --max_time_minutes 15 ^
  --max_evaluations 5000 ^
  --dt 0.008 ^
  --nphi 120 ^
  --nz 3 ^
  --workers 2 ^
  --chunk 300 ^
  --heading_step_deg 1.0 ^
  --speed_step 20.0 ^
  --t1_step 2.0 ^
  --gap12_step 2.0 ^
  --gap23_step 2.0 ^
  --fuse_step 2.0 ^
  --strict_validation ^
  --no_cd
echo Q3 Solver completed.
pause
