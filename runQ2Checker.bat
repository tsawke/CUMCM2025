@REM python Q2Check_v2.py --heading-deg 3.2 --speed 78 ^
@REM   --lhs 1024 --topk 16 --pat-iter 40 --workers auto ^
@REM   --dt-coarse 0.003 --nphi-coarse 360 --nz-coarse 7 ^
@REM   --dt-final 0.001 --nphi-final 720 --nz-final 11


python Q2Checker.py ^
  --heading-rad 0.122014 ^
  --speed 130 ^
  --drop-time 0.240000 ^
  --fuse 0.3 ^
  --dt 0.001 --nphi 960 --nz 13 ^
  --backend process --workers auto
