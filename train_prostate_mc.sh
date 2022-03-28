CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="ce" optim="adam" scheduler="step" wandb.project="unet-prostate_mc" train.max_epoch="100"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="ls" loss.alpha=0.05 optim="adam" scheduler="step" wandb.project="unet-prostate_mc" train.max_epoch="100"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" model.num_inp_channels="1" model.num_classes="3" loss="logit_margin" loss.margin="8" optim="adam" scheduler="step" wandb.project="unet-prostate_mc" train.max_epoch="100" 


#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" loss="focal" optim="adam" scheduler="step" wandb.project="unet-prostate_mc"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="prostate_mc" model="unet" loss="penalty_ent" optim="adam" scheduler="step" wandb.project="unet-prostate_mc"
