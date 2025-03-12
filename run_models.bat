@echo off

python ./src/extraction/auto_extract.py ./Dataset2

python ./src/scripts/Evaluation.py ./Dataset2

python ./src/scripts/copy_res_files.py ./Dataset2 ./res_ViT_SC_CBP test_result_ViT res_ViT \label


@REM powershell -Command "(Get-Content config.ini) -replace 'ViT-B/32', 'LongCLIP-L' | Set-Content config.ini"

@REM python ./src/extraction/auto_extract.py ./Dataset2

@REM python ./src/scripts/copy_res_files.py ./Dataset2 ./res_LongCLIP_SC_CBP test_result_LongCLIP res_LongCLIP \label

@REM python ./src/scripts/Evaluation.py ./res_LongCLIP_SC_CBP

@REM powershell -Command "(Get-Content config.ini) -replace 'LongCLIP-L', 'BAAI/BGE-VL-large' | Set-Content config.ini"

@REM python ./src/extraction/auto_extract.py ./Dataset2

@REM python ./src/scripts/copy_res_files.py ./Dataset2 ./res_BAAI_SC_CBP test_result_BAAI res_BAAI \label

@REM python ./src/scripts/Evaluation.py ./res_BAAI_SC_CBP

pause