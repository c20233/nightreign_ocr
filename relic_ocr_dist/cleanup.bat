@echo off
setlocal

echo ===================================================
echo   Night Reign OCR Analyzer - クリーンアップ (削除)
echo ===================================================
echo.
echo このバッチファイルは、ツールによって構築された仮想環境（venv）を
echo ディスク上から完全に削除し、クリーンアップします。
echo.
set /p CONFIRM="仮想環境を削除してもよろしいですか？ (y/n): "

if /i "%CONFIRM%"=="y" (
    echo.
    if exist "venv" (
        echo 仮想環境（venv）を削除しています...
        rmdir /s /q venv
        if exist "venv" (
            echo [エラー] 一部のファイルが使用中のため削除できませんでした。
            echo コマンドプロンプトや他のソフトでこのフォルダを開いていないか確認してください。
        ) else (
            echo 仮想環境の削除が完了しました。
        )
    ) else (
        echo 仮想環境（venv）は存在しません。
    )
    
    :: テンポラリPNGの削除
    echo 一時ファイルを削除しています...
    del /q _tmp_roi_*.png > nul 2>&1
    
    echo.
    echo クリーンアップが完了しました。
) else (
    echo.
    echo クリーンアップをキャンセルしました。
)

pause
