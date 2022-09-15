#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="ce" optim="adam" scheduler="step" wandb.project="unet-cardiac" wandb.enable=True
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="ce_dice" optim="adam" scheduler="step" wandb.project="unet-cardiac" wandb.enable=True
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="focal" optim="adam" scheduler="step" wandb.project="unet-cardiac" wandb.enable=True
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="penalty_ent" optim="adam" scheduler="step" wandb.project="unet-cardiac" wandb.enable=True
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="ls" optim="adam" scheduler="step" wandb.project="unet-cardiac"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="svls" optim="adam" scheduler="step" wandb.project="unet-cardiac"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="logit_margin" optim="adam" scheduler="step" wandb.project="unet-cardiac" wandb.enable=True
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="logit_margin" loss.margin="0" optim="adam" scheduler="step" wandb.project="unet-cardiac" wandb.enable=True
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="logit_margin" loss.margin="5" optim="adam" scheduler="step" wandb.project="unet-cardiac" wandb.enable=True
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="logit_margin" loss.margin="8" optim="adam" scheduler="step" wandb.project="unet-cardiac" wandb.enable=True
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="logit_margin" loss.margin="3" optim="adam" scheduler="step" wandb.project="unet-cardiac" wandb.enable=True
#
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="svls_logit_margin" loss.numclasses="4" optim="adam" scheduler="step" wandb.project="unet-cardiac"



#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="ls" loss.alpha="0.2" optim="adam" scheduler="step" wandb.project="unet-cardiac"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="ls" loss.alpha="0.3" optim="adam" scheduler="step" wandb.project="unet-cardiac"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="penalty_ent" loss.alpha="0.2" optim="adam" scheduler="step" wandb.project="unet-cardiac"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="penalty_ent" loss.alpha="0.3" optim="adam" scheduler="step" wandb.project="unet-cardiac"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="focal" loss.gamma="1" optim="adam" scheduler="step" wandb.project="unet-cardiac"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="focal" loss.gamma="2" optim="adam" scheduler="step" wandb.project="unet-cardiac"


#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="logit_margin_adaptive" optim="adam" scheduler="step" wandb.project="unet-cardiac"


##unet++
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unetpp" model.num_inp_channels="1" model.num_classes="4" loss="ce" optim="adam" scheduler="step" wandb.project="unet-cardiac"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unetpp" model.num_inp_channels="1" model.num_classes="4" loss="logit_margin" optim="adam" scheduler="step" wandb.project="unet-cardiac"



## save checkpoints in interval
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="ce" optim="adam" scheduler="step" wandb.project="unet-cardiac" wandb.enable=True train.keep_checkpoint_interval="10"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="logit_margin" loss.margin="5" optim="adam" scheduler="step" wandb.project="unet-cardiac" wandb.enable=True train.keep_checkpoint_interval="10"
CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="focal" optim="adam" scheduler="step" wandb.project="unet-cardiac" wandb.enable=True train.keep_checkpoint_interval="10"
