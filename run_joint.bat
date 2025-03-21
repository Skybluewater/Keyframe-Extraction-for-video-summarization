@echo off

@REM python ./src/extraction/auto_extract.py ./Dataset2 --joint attention --weights 0.5 0.5

@REM python ./src/scripts/Evaluation.py ./Dataset2

@REM python ./src/scripts/copy_res_files.py ./Dataset2 ./res_LongCLIP_SC_Att_0.5_0.5 test_result_LongCLIP res_LongCLIP \label


@REM python ./src/extraction/auto_extract.py ./Dataset2 --joint attention --weights 0.7 0.3

@REM python ./src/scripts/Evaluation.py ./Dataset2

@REM python ./src/scripts/copy_res_files.py ./Dataset2 ./res_LongCLIP_SC_Att_0.7_0.3 test_result_LongCLIP res_LongCLIP \label


python ./src/extraction/auto_extract.py ./Dataset2 --joint concatenate --threshold 0.5

python ./src/scripts/Evaluation.py ./Dataset2

python ./src/scripts/copy_res_files.py ./Dataset2 ./res_LongCLIP_SC_concat_0.5 test_result_LongCLIP res_LongCLIP labe


python ./src/extraction/auto_extract.py ./Dataset2 --joint CBP --threshold 0.5

python ./src/scripts/Evaluation.py ./Dataset2

python ./src/scripts/copy_res_files.py ./Dataset2 ./res_LongCLIP_SC_CBP_new_0.5 test_result_LongCLIP res_LongCLIP labe


python ./src/extraction/auto_extract.py ./Dataset2 --joint multiplication --threshold 0.5

python ./src/scripts/Evaluation.py ./Dataset2

python ./src/scripts/copy_res_files.py ./Dataset2 ./res_LongCLIP_SC_Mul_0.5 test_result_LongCLIP res_LongCLIP labe


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