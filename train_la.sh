CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="la" model="unet" model.num_classes="4" loss="ce" optim="adam" scheduler="step" wandb.project="unet-la" wandb.enable=True
CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="la" model="unet" model.num_classes="4" loss="ce_dice" optim="adam" scheduler="step" wandb.project="unet-la" wandb.enable=True
CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="la" model="unet" model.num_classes="4" loss="focal" optim="adam" scheduler="step" wandb.project="unet-la" wandb.enable=True
CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="la" model="unet" model.num_classes="4" loss="penalty_ent" optim="adam" scheduler="step" wandb.project="unet-la" wandb.enable=True
CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="la" model="unet" model.num_classes="4" loss="ls" optim="adam" scheduler="step" wandb.project="unet-la"
CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="la" model="unet" model.num_classes="4" loss="svls" optim="adam" scheduler="step" wandb.project="unet-la"
CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="la" model="unet" model.num_classes="4" loss="logit_margin" optim="adam" scheduler="step" wandb.project="unet-la" wandb.enable=True
CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="la" model="unet" model.num_classes="4" loss="logit_margin" loss.margin="0" optim="adam" scheduler="step" wandb.project="unet-la" wandb.enable=True
CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="la" model="unet" model.num_classes="4" loss="logit_margin" loss.margin="5" optim="adam" scheduler="step" wandb.project="unet-la" wandb.enable=True
CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="la" model="unet" model.num_classes="4" loss="logit_margin" loss.margin="8" optim="adam" scheduler="step" wandb.project="unet-la" wandb.enable=True
CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="la" model="unet" model.num_classes="4" loss="logit_margin" loss.margin="3" optim="adam" scheduler="step" wandb.project="unet-la" wandb.enable=True

