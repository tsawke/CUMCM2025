@echo off
echo 启动Numba并行优化版本...
echo 当前时间: %date% %time%
echo.

python solver5_numba_parallel.py

echo.
echo 优化完成时间: %date% %time%
pause
