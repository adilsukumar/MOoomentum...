@echo off
cd /d D:\22_SkinSense_ML
"D:\22_SkinSense_ML\.venv\Scripts\python.exe" "D:\22_SkinSense_ML\scripts\train.py" --arch efficientnetv2_rw_s --out models/effnet --epochs 18 --batch-size 32 --img-size 224 > "D:\22_SkinSense_ML\models\effnet_train_log2.txt" 2>&1
echo DONE_EXIT_%ERRORLEVEL% >> "D:\22_SkinSense_ML\models\effnet_train_log2.txt"
