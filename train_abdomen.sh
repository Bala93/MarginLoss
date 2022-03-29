CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="abdomen" data.batch_size="16" model="unet" model.num_inp_channels="1" model.num_classes="5" loss="ce" optim="adam" scheduler="step" wandb.project="unet-abdomen" train.max_epoch="100" log_period="100"
CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="abdomen" data.batch_size="16" model="unet" model.num_inp_channels="1" model.num_classes="5" loss="ls" optim="adam" scheduler="step" wandb.project="unet-abdomen" train.max_epoch="100" log_period="100"
CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="abdomen" data.batch_size="16" model="unet" model.num_inp_channels="1" model.num_classes="5" loss="logit_margin" optim="adam" scheduler="step" wandb.project="unet-abdomen" train.max_epoch="100" log_period="100"


#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="abdomen" model="unet" loss="focal" optim="adam" scheduler="step" wandb.project="unet-abdomen"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="abdomen" model="unet" loss="penalty_ent" optim="adam" scheduler="step" wandb.project="unet-abdomen"
