# Enhancing the Utilization of Uncertain Pixels in Semi-Supervised Semantic Segmentation

This is the code for the EUUP.

## Prepare Data

```
|-- ...
|--run.py
|--data
  |-- cityscapes  
  |-- VOCSegAug
    |-- index
    |-- image
    |-- label
```

## Train
```bash
torchrun --nproc_per_node=8 run.py train --config ./config/euup_deeplabv3p/voc_aug.yaml --num-gpus=8
```