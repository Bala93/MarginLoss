#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="ce" optim="adam" scheduler="step" wandb.project="unet-busi" optim.lr="0.0001"
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="ce_dice" optim="adam" scheduler="step" wandb.project="unet-busi" optim.lr="0.0001"
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="focal" optim="adam" scheduler="step" wandb.project="unet-busi" optim.lr="0.0001"
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="penalty_ent" optim="adam" scheduler="step" wandb.project="unet-busi" optim.lr="0.0001"
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="ls" optim="adam" scheduler="step" wandb.project="unet-busi"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="svls" loss.numclasses="2" optim="adam" scheduler="step" wandb.project="unet-busi" optim.lr="0.0001"
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="logit_margin" optim="adam" scheduler="step" wandb.project="unet-busi" optim.lr="0.0001"


#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="focal" loss.gamma="1" optim="adam" scheduler="step" wandb.project="unet-busi" wandb.enable=True optim.lr="0.0001"
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="focal" loss.gamma="2" optim="adam" scheduler="step" wandb.project="unet-busi" wandb.enable=True optim.lr="0.0001"
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="penalty_ent" loss.alpha="0.2" optim="adam" scheduler="step" wandb.project="unet-busi" wandb.enable=True  optim.lr="0.0001"
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="penalty_ent" loss.alpha="0.3" optim="adam" scheduler="step" wandb.project="unet-busi" wandb.enable=True optim.lr="0.0001"
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="ls" loss.alpha="0.2" optim="adam" scheduler="step" wandb.project="unet-busi" optim.lr="0.0001"
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="ls" loss.alpha="0.3" optim="adam" scheduler="step" wandb.project="unet-busi" optim.lr="0.0001"
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="logit_margin" loss.margin="5" optim="adam" scheduler="step" wandb.project="unet-busi" wandb.enable=True optim.lr="0.0001"
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="logit_margin" loss.margin="8" optim="adam" scheduler="step" wandb.project="unet-busi" wandb.enable=True optim.lr="0.0001"
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="logit_margin" loss.margin="3" optim="adam" scheduler="step" wandb.project="unet-busi" wandb.enable=True optim.lr="0.0001"

# CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="logit_margin" loss.margin="12" optim="adam" scheduler="step" wandb.project="unet-busi" wandb.enable=True optim.lr="0.0001"
# CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="logit_margin" loss.margin="15" optim="adam" scheduler="step" wandb.project="unet-busi" wandb.enable=True optim.lr="0.0001"


## seed 0
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="ce" optim="adam" scheduler="step" wandb.project="unet-busi" optim.lr="0.0001" seed=0
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="ce_dice" optim="adam" scheduler="step" wandb.project="unet-busi" optim.lr="0.0001" seed=0
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="focal" optim="adam" scheduler="step" wandb.project="unet-busi" optim.lr="0.0001" seed=0
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="penalty_ent" loss.alpha="0.3" optim="adam" scheduler="step" wandb.project="unet-busi" optim.lr="0.0001" seed=0
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="ls" loss.alpha="0.2" optim="adam" scheduler="step" wandb.project="unet-busi" optim.lr="0.0001" seed=0
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="svls" loss.numclasses="2" optim="adam" scheduler="step" wandb.project="unet-busi" optim.lr="0.0001" seed=0
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="logit_margin" loss.margin="12" optim="adam" scheduler="step" wandb.project="unet-busi" optim.lr="0.0001" seed=0

# seed 2
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="ce" optim="adam" scheduler="step" wandb.project="unet-busi" optim.lr="0.0001" seed=2
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="ce_dice" optim="adam" scheduler="step" wandb.project="unet-busi" optim.lr="0.0001" seed=2
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="focal" optim="adam" scheduler="step" wandb.project="unet-busi" optim.lr="0.0001" seed=2
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="penalty_ent" loss.alpha="0.3" optim="adam" scheduler="step" wandb.project="unet-busi" optim.lr="0.0001" seed=2
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="ls" loss.alpha="0.2" optim="adam" scheduler="step" wandb.project="unet-busi" optim.lr="0.0001" seed=2
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="svls" loss.numclasses="2" optim="adam" scheduler="step" wandb.project="unet-busi" optim.lr="0.0001" seed=2
#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="logit_margin" loss.margin="12" optim="adam" scheduler="step" wandb.project="unet-busi" optim.lr="0.0001" seed=2


#CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="logit_margin_l2" loss.margin="12" optim="adam" scheduler="step" wandb.project="unet-busi" optim.lr="0.0001" seed=2

CUDA_VISIBLE_DEVICES=5 python tools/train_net.py wandb.enable=True task="medseg" data="busi" model="unet" model.num_classes="2" loss="logit_margin_dice" loss.margin="12" optim="adam" scheduler="step" wandb.project="unet-busi" optim.lr="0.0001" seed=2
