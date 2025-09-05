@REM python Q2Solver_v6.py --algo hybrid --pop 64 --iter 60 --workers auto --topk 12 ^
@REM         --dt-coarse 0.002 --nphi-coarse 480 --nz-coarse 9 ^
@REM         --dt-final 0.0005 --nphi-final 960 --nz-final 13


@REM python Q2Solver_v7.py --algo all --pop 16 --iter 20 --workers auto --topk 6 ^
@REM   --dt-coarse 0.004 --nphi-coarse 240 --nz-coarse 7 ^
@REM   --dt-final 0.001 --nphi-final 480 --nz-final 9


@REM python Q2Solver_v7.py --algo all --pop 64 --iter 60 --workers auto --topk 12 ^
@REM   --dt-coarse 0.002 --nphi-coarse 480 --nz-coarse 9 ^
@REM   --dt-final 0.0005 --nphi-final 960 --nz-final 13

@REM python Q2Solver_v8.py --algo all --pop 16 --iter 20 --workers auto --topk 6 ^
@REM   --dt-coarse 0.004 --nphi-coarse 240 --nz-coarse 7 ^
@REM   --dt-final 0.001 --nphi-final 480 --nz-final 9

@REM python Q2Solver_v8.py --algo all --pop 64 --iter 60 --workers auto --topk 12 ^
@REM   --dt-coarse 0.002 --nphi-coarse 480 --nz-coarse 9 ^
@REM   --dt-final 0.0005 --nphi-final 960 --nz-final 13

@REM python Q2Solver_v9.py --algo sa --sa-iters 5000 --dt-coarse 0.004 --nphi-coarse 240 --nz-coarse 7

@REM python Q2Solver_v10.py ^
@REM   --algo all ^
@REM   --pop 16 --iter 20 --topk 8 --workers auto ^
@REM   --dt-coarse 0.006 --nphi-coarse 160 --nz-coarse 5 ^
@REM   --dt-final  0.002 --nphi-final  320 --nz-final  7 ^
@REM   --sa-iters 500 ^
@REM   --probe 128

python Q2Solver_v12.py ^
  --algo pattern ^
  --pop 64 --iter 60 --topk 12 --workers auto ^
  --dt-coarse 0.004 --nphi-coarse 240 --nz-coarse 7 ^
  --dt-final  0.001 --nphi-final  480 --nz-final  9 ^
  --sa-iters 2000 ^
  --probe 256
