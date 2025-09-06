python Q3Checker.py ^
  --excel result1_v4.xlsx ^
  --dt 0.002 --nphi 720 --nz 11 ^
  --backend thread --workers %NUMBER_OF_PROCESSORS% ^
  --chunk 3000 --block 8192 --margin 1e-12 --inner_log
