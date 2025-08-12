@echo off
REM 激活 Conda 環境
call conda activate my_env

REM 切換到工作目錄
cd C:\tim\aicam\main\fed_server\cloud_models

REM 執行 Python 腳本
python emqx_manager.py

REM 保持窗口打開以便查看輸出
pause