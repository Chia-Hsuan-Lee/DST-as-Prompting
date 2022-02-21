# DST-as-Prompting

This is the original implementation of "[Dialogue State Tracking with a Language Model using Schema-Driven Prompting](https://aclanthology.org/2021.emnlp-main.404/)" by [Chia-Hsuan Lee](https://chiahsuan156.github.io/), [Hao Cheng](https://sites.google.com/site/hcheng2site) and [Mari Ostendorf](https://people.ece.uw.edu/ostendorf/).

[**Installation**](#Installation) | [**Preprocess**](#Download-and-Preprocess-Data) | [**Training**](#Training) | [**Evaluation**](#Evaluation) | | [**Citation**](#Citation-and-Contact)

## Installation

```
$ conda create -n DST-prompt python=3.7
$ cd DST-as-Prompting
$ conda env update -n DST-prompt -f env.yml
```

To use Hugggingface seq2seq training scripts, install from source. 
```
pip install git+https://github.com/huggingface/transformers.git@2c2a31ffbcfe03339b1721348781aac4fc05bc5e
```

Clone the huggingface repository and checkout to specific commit. This is because the code is only tested on this version. 
You could also use your own seq2seq training script too!
```console
$ git clone https://github.com/huggingface/transformers.git
$ cd transformers
$ git checkout 2c2a31ffbcfe03339b1721348781aac4fc05bc5e
$ cd examples/pytorch/summarization/
$ pip install -r requirements.txt
```

## Download and Preprocess Data
Please download the data from MultiWOZ [github](https://github.com/budzianowski/multiwoz). 

```console
$ git clone https://github.com/budzianowski/multiwoz.git
```

`$DATA_DIR` will be `multiwoz/data/MultiWOZ_2.2`

```console
$ cd ~/DST-as-Prompting
$ python preprocess.py $DATA_DIR
```

## Training

```console
$ cd transformers
$ python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_predict \
    --train_file "$DATA_DIR/train.json" \
    --validation_file "$DATA_DIR/dev.json" \
    --test_file "$DATA_DIR/test.json" \
    --source_prefix "" \
    --output_dir /tmp/t5small_mwoz2.2 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
    --text_column="dialogue" \
    --summary_column="state" \
    --save_steps=500000
```

- `--model_name_or_path`: name of the model card, like `t5-small`, `t5-base`, etc

This should take ~32 hours to train on a single GPU. 
At the end of training, the model will get predictions on `$test_file` and store the results at `$output_dir/generated_predictions.txt` .

## Evaluation

```console
python postprocess.py --data_dir "$DATA_DIR" --out_dir "$DATA_DIR/dummy/" --test_idx "$DATA_DIR/test.idx" \
    --prediction_txt "$output_dir/generated_predictions.txt"

python eval.py --data_dir "$DATA_DIR" --prediction_dir "$DATA_DIR/dummy/" \
    --output_metric_file "$DATA_DIR/dummy/prediction_score"
```

## Citation and Contact

If you find our code or paper useful, please cite the paper:
```bib
@article{lee2021dialogue,
  title={Dialogue State Tracking with a Language Model using Schema-Driven Prompting},
  author={Lee, Chia-Hsuan and Cheng, Hao and Ostendorf, Mari},
  journal={arXiv preprint arXiv:2109.07506},
  year={2021}
}
```

Please contact Chia-Hsuan Lee (chiahlee[at]uw.edu) for questions and suggestions.
