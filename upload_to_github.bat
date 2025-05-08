@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

echo ===================================
echo 上傳專案到GitHub倉庫
echo ===================================
echo.

:: 檢查Git是否已安裝
git --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [錯誤] 未檢測到Git，請先安裝Git。
    echo 您可以從 https://git-scm.com/downloads 下載Git。
    pause
    exit /b 1
)

echo [信息] 檢測到Git環境，開始上傳專案...
echo.

:: 設置倉庫URL
set REPO_URL=https://github.com/aken1023/langchain_faiss_deepseek_app.git

:: 檢查是否已初始化Git倉庫
if not exist ".git" (
    echo [信息] 初始化Git倉庫
    git init
    
    :: 設置Git用戶信息
    git config user.email "aken1023@gmail.com"
    git config user.name "aken1023"
)

:: 設置遠程倉庫
echo [信息] 檢查遠程倉庫
git remote -v | findstr "origin" > nul 2>&1
if %errorlevel% neq 0 (
    echo [信息] 添加遠程倉庫
    git remote add origin %REPO_URL%
) else (
    echo [信息] 更新遠程倉庫URL
    git remote set-url origin %REPO_URL%
)

:: 檢查敏感信息
echo [信息] 檢查敏感信息
echo [警告] GitHub會阻止包含API密鑰等敏感信息的推送。
echo [警告] 請確保.env和代碼中沒有真實API密鑰

echo [信息] 創建.gitignore文件以排除敏感文件
(
  echo # 敏感文件
  echo .env
  echo __pycache__/
  echo *.pyc
  echo .DS_Store
) > .gitignore

:: 添加所有文件
echo [信息] 添加文件到Git
git add .

:: 提交更改
echo [信息] 提交更改
set /p COMMIT_MSG=請輸入提交訊息 (預設: "更新專案"): 
if "%COMMIT_MSG%"=="" set COMMIT_MSG=更新專案

git commit -m "%COMMIT_MSG%"

:: 清理歷史提交中的敏感信息
echo [信息] 檢查是否需要清理歷史提交
echo [警告] GitHub檢測到歷史提交中可能包含API密鑰
echo [信息] 嘗試重置提交歷史

:: 創建新的初始提交
git update-ref -d HEAD

:: 重新添加和提交文件
git add .
git commit -m "%COMMIT_MSG%"

:: 推送到GitHub
echo [信息] 推送到GitHub
echo [信息] 您可能需要輸入GitHub帳號和密碼或個人訪問令牌(PAT)

:: 檢查是否有main分支
git show-ref --verify --quiet refs/heads/main
if %errorlevel% neq 0 (
    echo [信息] 創建main分支
    git branch -M main
)

:: 推送到遠程倉庫
git push -u origin main

set PUSH_STATUS=%errorlevel%
if %PUSH_STATUS% equ 0 (
    echo.
    echo ===================================
    echo 專案已成功上傳到GitHub！
    echo 倉庫URL: %REPO_URL%
    echo ===================================
) else (
    echo.
    echo [錯誤] 上傳失敗，請檢查錯誤訊息。
    echo 可能的原因:
    echo 1. 網絡連接問題
    echo 2. GitHub認證失敗
    echo 3. 遠程倉庫已有內容，需要先pull
    echo.
    echo 如果是認證問題，請考慮使用個人訪問令牌(PAT):
    echo 1. 訪問 https://github.com/settings/tokens 創建令牌
    echo 2. 使用令牌作為密碼登入
    echo.
    echo 如果遠程倉庫已有內容，請嘗試:
    echo git pull --rebase origin main
    echo 然後再次運行此腳本
)


pause