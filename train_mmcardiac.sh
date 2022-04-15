CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="mmcardiac" model="unet" model.num_classes="4" loss="ce" optim="adam" scheduler="step" wandb.project="unet-mmcardiac"
CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="mmcardiac" model="unet" model.num_classes="4" loss="ls" optim="adam" scheduler="step" wandb.project="unet-mmcardiac"
CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="mmcardiac" model="unet" model.num_classes="4" loss="logit_margin" optim="adam" scheduler="step" wandb.project="unet-mmcardiac"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="focal" optim="adam" scheduler="step" wandb.project="unet-cardiac"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="penalty_ent" optim="adam" scheduler="step" wandb.project="unet-cardiac"
#
