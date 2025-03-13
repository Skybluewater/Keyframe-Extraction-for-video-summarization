@echo off


python .\src\extraction\Redundancy.py ./res_LongCLIP_SC_concat --threshold 0.5

python ./src/scripts/Evaluation.py ./res_LongCLIP_SC_concat --threshold 0.5



python .\src\extraction\Redundancy.py ./res_LongCLIP_SC_CBP_new --threshold 0.5

python ./src/scripts/Evaluation.py ./res_LongCLIP_SC_CBP_new --threshold 0.5



python .\src\extraction\Redundancy.py ./res_LongCLIP_SC_Att_0.5_0.5 --threshold 0.5

python ./src/scripts/Evaluation.py ./res_LongCLIP_SC_Att_0.5_0.5 --threshold 0.5



python .\src\extraction\Redundancy.py ./res_LongCLIP_SC_Att_0.7_0.3 --threshold 0.5

python ./src/scripts/Evaluation.py ./res_LongCLIP_SC_Att_0.7_0.3 --threshold 0.5



python .\src\extraction\Redundancy.py ./res_LongCLIP_SC_sub --threshold 0.5

python ./src/scripts/Evaluation.py ./res_LongCLIP_SC_sub --threshold 0.5



python ./src/extraction/auto_extract.py ./Dataset2 --joint multiplication --threshold 0.5

python ./src/scripts/Evaluation.py ./Dataset2

python ./src/scripts/copy_res_files.py ./Dataset2 ./res_LongCLIP_SC_Mul test_result_LongCLIP res_LongCLIP labe cli


@REM python ./src/extraction/auto_extract.py ./Dataset2 --joint minus

@REM python ./src/scripts/Evaluation.py ./Dataset2

@REM python ./src/scripts/copy_res_files.py ./Dataset2 ./res_LongCLIP_SC_sub test_result_LongCLIP res_LongCLIP \label





@REM python ./src/extraction/auto_extract_subtract_attention_0.5_0.5.py ./Dataset2

@REM python ./src/scripts/Evaluation.py ./res_ViT_SC_CBP

@REM python ./src/scripts/copy_res_files.py ./Dataset2 ./res_LongCLIP_SC_Att_0.5_0.5 test_result_LongCLIP res_LongCLIP \label

@REM powershell -Command "(Get-Content config.ini) -replace 'ViT-B/32', 'LongCLIP-L' | Set-Content config.ini"

@REM python ./src/extraction/auto_extract.py ./Dataset2

@REM python ./src/scripts/copy_res_files.py ./Dataset2 ./res_LongCLIP_SC_CBP test_result_LongCLIP res_LongCLIP \label

@REM python ./src/scripts/Evaluation.py ./res_LongCLIP_SC_CBP

@REM powershell -Command "(Get-Content config.ini) -replace 'LongCLIP-L', 'BAAI/BGE-VL-large' | Set-Content config.ini"

@REM python ./src/extraction/auto_extract.py ./Dataset2

@REM python ./src/scripts/copy_res_files.py ./Dataset2 ./res_BAAI_SC_CBP test_result_BAAI res_BAAI \label

@REM python ./src/scripts/Evaluation.py ./res_BAAI_SC_CBP

pause