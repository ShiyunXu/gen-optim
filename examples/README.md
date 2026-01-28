# Examples with GeN

## Time-Series Modeling
We provide a script in `time_series.py` that demonstrates how to use GeN optimizer for time-series regression tasks with 1D-CNN and Vanilla Transformer models (~5M parameters). This example shows how to replace AdamW with GeN for time-series forecasting.

```bash
# Train 1D-CNN with GeN optimizer
python time_series.py --model cnn1d --dataset synthetic --epochs 50 --lr_scheduler GeN --lazy_freq 4

# Train Vanilla Transformer with GeN optimizer
python time_series.py --model transformer --dataset synthetic --epochs 50 --lr_scheduler GeN --lazy_freq 4

# Compare with AdamW (without GeN)
python time_series.py --model cnn1d --dataset synthetic --epochs 50 --lr_scheduler cosine --optim adamw
```

Key arguments:

* `--model`: Model architecture, either `cnn1d` (1D-CNN) or `transformer` (Vanilla Transformer).
* `--dataset`: Dataset to use, currently supports `synthetic` for demonstration.
* `--seq_len`: Time-series sequence length; default is 100.
* `--train_samples`: Number of training samples; default is 10000.
* `--test_samples`: Number of test samples; default is 2000.
* `--lr`: Learning rate; default is 1e-4.
* `--bs`: Logical batch size; default is 256.
* `--mini_bs`: Physical batch size for gradient accumulation; default is 64.
* `--epochs`: Number of epochs; default is 50.
* `--optim`: Base optimizer to use (`sgd` or `adamw`). For GeN, use base optimizers like adamw.
* `--lazy_freq`: How often to update the learning rate with GeN; default is 4. Higher values update less frequently, saving training time.
* `--lr_scheduler`: Learning rate scheduler. Use `GeN` for GeN, also supports `cosine`, `multistep`, and `none`.
* `--seed`: Random seed for reproducibility.

**Note**: This example uses synthetic time-series data. For real-world applications, replace the `TimeSeriesDataset` class with your own dataset loader.

## Image classification
We provide scripts in `cv.py` that train any [TIMM model](https://github.com/huggingface/pytorch-image-models/tree/main/timm/models) on image classification datasets such as CIFAR10, CIFAR100, SVHN, ImageNet, Places365, INaturalist, etc. This script can test other learning-rate-free methods and heuristic learning rate schedules.
```plaintext
python cv.py --model resnet18 --dataset_name CIFAR10 --lazy_freq 4 --lr_scheduler GeN --epochs 5 --bs 500
```

Key arguments:

* `--dataset_name`: Datasets to train on, e.g. SVHN, CIFAR10 (default), CIFAR100, Food101, GTSRB. New dataset can be easily added.

* `--model`: The pretrained model from TIMM, check the full list by `timm.list_models(pretrained=True)`. Default is `vit_base_patch16_224`

* `--lr`: Learning rate, default is 1e-4. This is initial learning rate for GeN.

* `--mini_bs` : Physical batch size for gradient accumulation that determines memory and speed, but not accuracy; default is 50.

* `--bs` : Logical batch size that determines the convergence and accuracy, should be multiple of `physical_batch_size`; default is 500.

* `--epochs`: Number of epochs; default is 5.

* `--optim`: Optimizers to use. One of `sgd`,`adamw`,`prodigy`,`dadaptadam`,`dadaptsgd`,`dog`. For GeN, use base optimizers like sgd and adamw.

* `--lazy_freq`: How often to update the learning rate; default is 4. Higher values updates less frequently, saving training time but may slow convergence if training is not sufficiently long.

* `--lr_scheduler`: Learning rate schedules. One of `cosine`, `multistep` and `GeN` (ours).

## NLU: finetuning RoBERTa with GeN

We finetune RoBERTa in `NLU/` folder, which is adapted from [LoRA codebase](https://github.com/microsoft/LoRA/tree/main/examples/NLU). Here is an example run using the same setup as original LoRA training. 
```
python run_glue_no_trainer_auto.py \
  --model_name_or_path roberta-base \
  --task_name sst2 \
  --max_length 512 \
  --per_device_train_batch_size 128 \
  --lr 2e-5 \
  --num_train_epochs 10 \
  --lr_scheduler_type GeN\
  --lazy_freq 8 \
  --seed 0
```
Here we can adjust to bitfit/lora/FMT (how), match the hyperparameters in paper & colab!!

Key arguments:

* `--task_name`: GLUE datasets to train on, e.g. sst2, qnli, mrpc, cola, etc.
* `--model_name_or_path`: The pretrained model from Huggingface.
* `--lr_scheduler_type`: Use 'GeN' for GeN. Default is `linear`, also supporting cosine and constant.
* `--per_device_train_batch_size`: Batch size per device.
* `--num_train_epochs`: Number of epochs.
* `--lr`: Learning rate.
* `--lazy_freq`: How often to update the learning rate with GeN.
* `--seed`: Random seed for reproducibility.

## NLG: finetuning GPT2 with GeN
We finetune GPT2 with LoRA in `NLG/` folder, which is adapted from [LoRA codebase](https://github.com/microsoft/LoRA/tree/main/examples/NLG). 
Here is an example run using the same hyperparameters as original LoRA training (the only new hyperparameter is `lazy_freq`). Note `scheduler=dev_perf` prohibits GeN to be overridden.
```
python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_autoft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --dataset_name e2e\
    --train_batch_size 256 \
    --valid_batch_size 64 \
    --seq_len 128 \
    --model_card gpt2.sm \
    --init_checkpoint pretrained_checkpoints/gpt2-pytorch_model.bin \
    --platform local \
    --lr 1e-3 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler dev_perf \
    --warmup_step 500 \
    --max_epoch 5 \
    --eval_interval 100 \
    --lora_dim 4 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir /to_be_added \
    --random_seed 110 \
    --lazy_freq 4
```

Key arguments:

* `--train_data`, `--valid_data`: Paths to training and validation data.
* `--dataset_name`: Dataset name, e.g. e2e, dart, webnlg.
* `--train_batch_size`, `--valid_batch_size`: Batch sizes for training and validation.
* `--seq_len`: Sequence length.
* `--model_card`: Model configuration, e.g. gpt2.sm.
* `--init_checkpoint`: Path to pretrained checkpoint.
* `--lr`: Learning rate.
* `--scheduler`: Use `dev_perf` for GeN.
* `--max_epoch`: Number of epochs.
* `--lora_dim`, `--lora_alpha`, `--lora_dropout`: LoRA parameters.
* `--label_smooth`: Label smoothing.
* `--lazy_freq`: How often to update the learning rate with GeN.
* `--random_seed`: Random seed for reproducibility.
  
To run the evaluation, you must run `bash download_evalscript.sh` first in `NLG/eval/` folder.

## Recommendation system
We build on [BERT4Rec-VAE-Pytorch](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch). To apply GeN, we replace `trainers/base.py` with our `recommendation_base.py`. Example run: 
```bash
python main.py --template train_bert --auto_lr 1
```

Key arguments:

* `--template`: Training template, e.g. `train_bert`.
* `--auto_lr`: Set to 1 to enable GeN learning rate adaptation.
* `--epochs`: Number of training epochs.
* `--batch_size`: Batch size for training.
* `--lr`: Learning rate.
* `--seed`: Random seed for reproducibility.

## Object detection/Instance segmentation
We build on [TorchVision Object Detection Finetuning Tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html). To apply GeN, we replace `engine.py` with our `dect_segm_engine.py`.

Key arguments:

* `--model`: Model architecture, e.g. `fasterrcnn_resnet50_fpn`.
* `--dataset`: Dataset to use, e.g. COCO, VOC.
* `--epochs`: Number of epochs.
* `--batch_size`: Batch size for training.
* `--lr`: Learning rate.
* `--lazy_freq`: How often to update the learning rate with GeN.
* `--seed`: Random seed for reproducibility.
