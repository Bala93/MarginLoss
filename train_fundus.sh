CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="fundus" data.batch_size="4" model="unet" model.num_inp_channels="3" model.num_classes="3" loss="ce" optim="adam" scheduler="step" wandb.project="unet-fundus" train.max_epoch="100" log_period="100"
# CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="fundus" data.batch_size="16" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="ls" optim="adam" scheduler="step" wandb.project="unet-fundus" train.max_epoch="100" log_period="100"
# CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="fundus" data.batch_size="16" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="logit_margin" optim="adam" scheduler="step" wandb.project="unet-fundus" train.max_epoch="100" log_period="100"


#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="fundus" model="unet" loss="focal" optim="adam" scheduler="step" wandb.project="unet-fundus"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="fundus" model="unet" loss="penalty_ent" optim="adam" scheduler="step" wandb.project="unet-fundus"
