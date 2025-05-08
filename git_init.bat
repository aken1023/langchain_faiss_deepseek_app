@echo off
chcp 950 > nul
REM 初始化 Git 倉庫
echo 開始初始化 Git 倉庫...
git init

REM （可選）建立 .gitignore 檔案
echo # 請在此添加要忽略的檔案或資料夾 > .gitignore
echo 已建立 .gitignore 檔案（如需請自行修改內容）。

REM 顯示完成訊息
echo Git 倉庫初始化完成！
echo 請按任意鍵結束...
pause > nul
 
