#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="ce" optim="adam" scheduler="step" wandb.project="unet-prostate_mc" 
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="ce_dice" optim="adam" scheduler="step" wandb.project="unet-prostate_mc" train.max_epoch="100" 
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="focal" optim="adam" scheduler="step" wandb.project="unet-prostate_mc" 
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="penalty_ent" optim="adam" scheduler="step" wandb.project="unet-prostate_mc" 
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="ls" optim="adam" scheduler="step" wandb.project="unet-prostate_mc" 
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="svls" loss.numclasses="3" optim="adam" scheduler="step" wandb.project="unet-prostate_mc" 
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="logit_margin" optim="adam" scheduler="step" wandb.project="unet-prostate_mc" 
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="logit_margin" loss.margin="5" optim="adam" scheduler="step" wandb.project="unet-prostate_mc" train.max_epoch="100" 
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="logit_margin" loss.margin="4" optim="adam" scheduler="step" wandb.project="unet-prostate_mc" train.max_epoch="100" 
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="logit_margin" loss.margin="3" optim="adam" scheduler="step" wandb.project="unet-prostate_mc" train.max_epoch="100" 
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="logit_margin" loss.margin="2" optim="adam" scheduler="step" wandb.project="unet-prostate_mc" train.max_epoch="100" 
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="logit_margin" loss.margin="1" optim="adam" scheduler="step" wandb.project="unet-prostate_mc" train.max_epoch="100" 
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="logit_margin" loss.margin="0" optim="adam" scheduler="step" wandb.project="unet-prostate_mc" train.max_epoch="100" 

CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="ls" loss.alpha="0.2" optim="adam" scheduler="step" wandb.project="unet-prostate_mc" 
CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="ls" loss.alpha="0.3" optim="adam" scheduler="step" wandb.project="unet-prostate_mc" 
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="logit_margin" loss.margin="8" optim="adam" scheduler="step" wandb.project="unet-prostate_mc" train.max_epoch="100" 
CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="focal" loss.gamma="1" optim="adam" scheduler="step" wandb.project="unet-prostate_mc" 
CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="focal" loss.gamma="2" optim="adam" scheduler="step" wandb.project="unet-prostate_mc" 
CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="penalty_ent" loss.alpha="0.2" optim="adam" scheduler="step" wandb.project="unet-prostate_mc" 
CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="penalty_ent" loss.alpha="0.3" optim="adam" scheduler="step" wandb.project="unet-prostate_mc" 
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="svls_logit_margin" loss.numclasses="3" optim="adam" scheduler="step" wandb.project="unet-prostate_mc" train.max_epoch="100" 
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="logit_margin_adaptive" optim="adam" scheduler="step" wandb.project="unet-prostate_mc" 
