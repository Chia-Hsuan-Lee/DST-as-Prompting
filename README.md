# DST-as-Prompting
This includes source code of Chia-Hsuan Lee, Hao Cheng, Mari Ostendorf. "[Dialogue State Tracking with a Language Model using Schema-Driven Prompting][paper]" 2021.

If you find our code or paper useful, please cite the paper:
```
@article{lee2021dialogue,
  title={Dialogue State Tracking with a Language Model using Schema-Driven Prompting},
  author={Lee, Chia-Hsuan and Cheng, Hao and Ostendorf, Mari},
  journal={arXiv preprint arXiv:2109.07506},
  year={2021}
}
```

## Content

1. [Installation](#installation)
2. [Download & Preprocess Data](#download-and-preprocess-data)
3. [Prompt-based DST](#prompt-based-DST)


## Installation

```
$ conda create -n DST-prompt python=3.6.13
$ conda env update -n DST-prompt -f ./transformers/t5-DST.yml
```

## Download and Preprocess Data
Please download the data from MultiWOZ [github](https://github.com/budzianowski/multiwoz). 
