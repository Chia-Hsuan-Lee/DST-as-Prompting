CUDA_VISIBLE_DEVICES=1 python examples/seq2seq/run_seq2seq.py \
    --model_name_or_path t5-base \
    --do_train \
    --do_predict \
    --task translation_source_to_target \
    --source_prefix "get dialogue state: " \
    --train_file "/home/t-clee/MultiWoz2.2/train_WOZ_description_PVs.json"\
    --validation_file "/home/t-clee/MultiWoz2.2/dev_WOZ_description_PVs.json"\
    --test_file "/home/t-clee/MultiWoz2.2/test_WOZ_description_PVs.json"\
    --output_dir /tmp/WOZ_description_PVs\
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
    --save_steps 100000 \
    --eval_steps 100000 \
    --logging_steps 100000 
    #--overwrite_output_dir \
    #--do_eval \
