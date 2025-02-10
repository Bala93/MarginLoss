### *Calibrating Segmentation Networks with Margin-based Label Smoothing* 
[MedIA 2023](https://arxiv.org/abs/2209.09641)

### *Trust your neighbours: Penalty-based constraints for model calibration* 
[MICCAI 2023](https://arxiv.org/abs/2303.06268)

### *Neighbor-Aware Calibration of Segmentation Networks with Penalty-Based Constraints* 
[MedIA 2024](https://arxiv.org/abs/2401.14487) 



## Examples
### CRaC
[CRaC](https://github.com/Bala93/CRac)
### NACL
```
python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="adaptive_margin_svls" loss.kernel_ops="mean" optim="adam" scheduler="step" wandb.project="unet-cardiac" loss.is_margin=True
```

Implementation of NACL can also be found in [MONAI](https://docs.monai.io/en/latest/losses.html#naclloss)



### MbLS
```
python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" data.ratio=0.5 model="unet" model.num_classes="4" loss="logit_margin" loss.margin="5" optim="adam" scheduler="step" wandb.project="unet-cardiac"
```

## Citations
```
@article{murugesan2024neighbor,
  title={Neighbor-Aware Calibration of Segmentation Networks with Penalty-Based Constraints},
  author={Murugesan, Balamurali and Vasudeva, Sukesh Adiga and Liu, Bingyuan and Lombaert, Herv{\'e} and Ayed, Ismail Ben and Dolz, Jose},
  journal={arXiv preprint arXiv:2401.14487},
  year={2024}
}
```
```
@article{murugesan2022calibrating,
  title={Calibrating Segmentation Networks with Margin-based Label Smoothing},
  author={Murugesan, Balamurali and Liu, Bingyuan and Galdran, Adrian and Ayed, Ismail Ben and Dolz, Jose},
  journal={Medical Image Analysis (MedIA)},
  year={2022}
}
```

```
@article{murugesan2023trust,
  title={Trust your neighbours: Penalty-based constraints for model calibration},
  author={Murugesan, Balamurali and Adiga V, Sukesh and Liu, Bingyuan and Lombaert, Herv{\'e} and Ayed, Ismail Ben and Dolz, Jose},
  journal={Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2023}
}
```
