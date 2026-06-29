@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo   Night Reign OCR Analyzer - セットアップ ＆ 実行
echo ===================================================
echo.

:: 1. Pythonの存在チェック
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [エラー] Pythonがシステムにインストールされていないか、PATHが通っていません。
    echo Python 3.10～3.12（64-bit）をインストールし、「Add Python to PATH」にチェックを入れてから再実行してください。
    pause
    exit /b 1
)

:: 2. 仮想環境の作成
if not exist "venv" (
    echo 仮想環境（venv）を作成しています...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [エラー] 仮想環境の作成に失敗しました。
        pause
        exit /b 1
    )
    echo 仮想環境の作成が完了しました。
    echo.
)

:: 3. 仮想環境のアクティベート
call venv\Scripts\activate.bat

:: 4. pipのアップグレード
echo pipをアップグレードしています...
python -m pip install --upgrade pip > nul
echo.

:: 5. 実行モードの選択（初回またはパッケージ未インストール時）
if not exist "venv\.installed" (
    echo インストールするバージョンを選択してください：
    echo [1] CPU版 (GPUを搭載していないPC、または他社製GPU用)
    echo [2] GPU版 (NVIDIA GeForce RTXなどのグラフィックボード搭載PC用)
    echo.
    set /p CHOICE="番号を入力してEnterを押してください (1 または 2): "

    if "!CHOICE!"=="2" (
        echo.
        echo GPU版をインストールしています（大容量のダウンロードが発生します）...
        pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
    ) else (
        echo.
        echo CPU版をインストールしています...
        pip install paddlepaddle
    )

    echo.
    echo その他の必要パッケージをインストールしています...
    pip install -r requirements.txt

    if %errorlevel% equ 0 (
        echo. > venv\.installed
        echo インストールが正常に完了しました！
    ) else (
        echo [エラー] パッケージのインストール中にエラーが発生しました。
        pause
        exit /b 1
    )
    echo.
)

:: 6. 動画ファイルの存在確認
if not exist "game_play_record.mp4" (
    echo [警告] 'game_play_record.mp4' が見つかりません。
    echo このフォルダに解析したい動画ファイル（game_play_record.mp4）を配置してから、
    echo 何かキーを押すと解析を開始します。
    echo.
    pause
)

:: 7. スクリプトの実行
echo.
echo OCR解析を実行します...
python main.py
echo.
echo 解析が完了しました。出力された 'game_stats.json' をご確認ください。
pause
