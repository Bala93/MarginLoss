#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="kidney" data.batch_size="16" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="ce" optim="adam" scheduler="step" wandb.project="unet-kidney" train.max_epoch="100" log_period="100"
CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="kidney" data.batch_size="16" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="ls" optim="adam" scheduler="step" wandb.project="unet-kidney" train.max_epoch="100" log_period="100"
CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="kidney" data.batch_size="16" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="logit_margin" optim="adam" scheduler="step" wandb.project="unet-kidney" train.max_epoch="100" log_period="100"


#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="kidney" model="unet" loss="focal" optim="adam" scheduler="step" wandb.project="unet-kidney"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="kidney" model="unet" loss="penalty_ent" optim="adam" scheduler="step" wandb.project="unet-kidney"
