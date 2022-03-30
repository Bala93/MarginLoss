CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="brainatlas" data.batch_size="4" model="unet" model.num_classes="5" model.num_inp_channels="3" optim="adam" scheduler="step" wandb.project="unet-brainatlas" log_period="50" train.max_epoch="100"
CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="brainatlas" data.batch_size="4" model="unet" model.num_classes="5" model.num_inp_channels="3" loss="ls" optim="adam" scheduler="step" wandb.project="unet-brainatlas" log_period="50" train.max_epoch="100"
CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="brainatlas" data.batch_size="4" model="unet" model.num_classes="5" model.num_inp_channels="3" loss="logit_margin" optim="adam" scheduler="step" wandb.project="unet-brainatlas" log_period="50" train.max_epoch="100" 

#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="focal" optim="adam" scheduler="step" wandb.project="unet-cardiac"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="penalty_ent" optim="adam" scheduler="step" wandb.project="unet-cardiac"
C
