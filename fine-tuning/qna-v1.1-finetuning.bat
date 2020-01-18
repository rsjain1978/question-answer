set BASE_DIR=C:\MachineLearning\repos\personel\qna-wrkspace\question-answer
set SQUAD_DIR=%BASE_DIR%\data\v1.1
set BASE_MODEL_DIR=%BASE_DIR%\base-models
set OUTPUT_MODEL_DIR=%BASE_DIR%\outputs

REM python run_squad.py --model_type distilbert --model_name_or_path distilbert-base-uncased-distilled-squad --do_train --do_eval --do_lower_case --train_file %SQUAD_DIR%\train\train-v1.1.json --predict_file %SQUAD_DIR%\validate\dev-v1.1.json --per_gpu_train_batch_size 12 --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --overwrite_output_dir --output_dir %OUTPUT_MODEL_DIR%\distilbert-base-uncased-distilled-squad-finetuned

python run_squad.py --model_type distilbert --model_name_or_path %BASE_MODEL_DIR%\distilbert-base-uncased-distilled-squad --do_train --do_eval --do_lower_case --train_file %SQUAD_DIR%\train\train-v1.1.json --predict_file %SQUAD_DIR%\validate\dev-v1.1.json --per_gpu_train_batch_size 12 --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --overwrite_output_dir --output_dir %OUTPUT_MODEL_DIR%\distilbert-base-uncased-distilled-squad-finetuned