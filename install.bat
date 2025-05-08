@echo off
echo ===================================
echo 知識庫問答系統 - 安裝腳本
echo ===================================
echo.

:: 檢查Python是否已安裝
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [錯誤] 未檢測到Python，請先安裝Python 3.8或更高版本。
    echo 您可以從 https://www.python.org/downloads/ 下載Python。
    pause
    exit /b 1
)

echo [信息] 檢測到Python環境，開始安裝依賴項...
echo.

:: 創建虛擬環境（可選）
echo [信息] 是否創建虛擬環境? (推薦)
echo 1. 是 - 創建名為'venv'的虛擬環境
echo 2. 否 - 直接在當前Python環境中安裝
set /p create_venv="請選擇 (1/2): "

if "%create_venv%"=="1" (
    echo [信息] 正在創建虛擬環境...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [錯誤] 虛擬環境創建失敗，請確保已安裝venv模塊。
        echo 嘗試運行: pip install virtualenv
        pause
        exit /b 1
    )
    echo [信息] 虛擬環境創建成功，正在激活...
    call venv\Scripts\activate
    echo [信息] 虛擬環境已激活: %VIRTUAL_ENV%
)

:: 升級pip
echo [信息] 正在升級pip...
python -m pip install --upgrade pip

:: 安裝依賴項
echo [信息] 正在安裝依賴項...
pip install -r requirements.txt

:: 檢查CUDA環境
echo [信息] 正在檢查CUDA環境...
nvcc --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] 未檢測到CUDA環境，GPU加速可能無法使用。
    echo [信息] 您可以從NVIDIA官網下載並安裝CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
    echo [信息] 繼續安裝，但將使用CPU模式...
    
    :: 如果沒有CUDA，替換為CPU版本
    echo [信息] 將faiss-gpu替換為faiss-cpu...
    powershell -Command "(Get-Content requirements.txt) -replace 'faiss-gpu==1.7.4', 'faiss-cpu==1.7.4' | Set-Content requirements.txt"
) else (
    echo [信息] 檢測到CUDA環境，將使用GPU加速！
)

:: 檢查PyTorch安裝
echo [信息] 正在檢查PyTorch安裝...
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())" > nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] PyTorch安裝可能有問題，嘗試單獨安裝...
    pip uninstall -y torch torchvision torchaudio
    
    :: 根據CUDA環境選擇安裝命令
    nvcc --version > nul 2>&1
    if %errorlevel% neq 0 (
        echo [信息] 安裝CPU版本的PyTorch...
        pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cpu
    ) else (
        echo [信息] 安裝支持CUDA的PyTorch...
        pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118
    )
    
    python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())" > nul 2>&1
    if %errorlevel% neq 0 (
        echo [錯誤] PyTorch安裝失敗，請嘗試手動安裝。
        echo 可能需要安裝Visual C++ Redistributable。
        echo 請訪問: https://pytorch.org/get-started/locally/ 獲取更多信息。
    ) else (
        echo [信息] PyTorch安裝成功！
        python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"
    )
) else (
    echo [信息] PyTorch已正確安裝！
    python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"
)

:: 檢查其他關鍵依賴
echo [信息] 正在檢查其他關鍵依賴...

:: 檢查faiss
python -c "import faiss; print('FAISS版本:', faiss.__version__); print('是否支持GPU:', hasattr(faiss, 'GpuIndexFlatL2'))" > nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] FAISS安裝可能有問題，請檢查上方錯誤信息。
) else (
    python -c "import faiss; print('[信息] FAISS版本:', faiss.__version__); print('[信息] 是否支持GPU:', 'Yes' if hasattr(faiss, 'GpuIndexFlatL2') else 'No')"
)

:: 檢查其他依賴
python -c "import langchain, transformers, tiktoken; print('[信息] LangChain, Transformers, Tiktoken已正確安裝')" > nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] 某些關鍵依賴可能未正確安裝，請檢查上方錯誤信息。
) else (
    echo [信息] 所有關鍵依賴已正確安裝！
)

:: 顯示GPU信息（如果可用）
python -c "import torch; print('[信息] GPU數量:', torch.cuda.device_count()); [print(f'[信息] GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print('[信息] 未檢測到可用的GPU')" 2>nul

echo.
echo ===================================
echo 安裝完成！
echo.
echo 運行系統:
echo python langchain_faiss_deepseek_app.py
echo ===================================

if "%create_venv%"=="1" (
    echo 注意: 每次運行前需要先激活虛擬環境:
    echo call venv\Scripts\activate
)

pause