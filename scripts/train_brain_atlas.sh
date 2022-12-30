CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="brainatlas" data.batch_size="4" model="unet" model.num_classes="4" model.num_inp_channels="3" loss="ce" optim="adam" scheduler="step" wandb.project="unet-brainatlas" log_period="50" train.max_epoch="100"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="brainatlas" data.batch_size="4" model="unet" model.num_classes="4" model.num_inp_channels="3" loss="ce_dice" optim="adam" scheduler="step" wandb.project="unet-brainatlas" log_period="50" train.max_epoch="100"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="brainatlas" data.batch_size="4" model="unet" model.num_classes="4" model.num_inp_channels="3" loss="focal" optim="adam" scheduler="step" wandb.project="unet-brainatlas"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="brainatlas" data.batch_size="4" model="unet" model.num_classes="4" model.num_inp_channels="3" loss="penalty_ent" optim="adam" scheduler="step" wandb.project="unet-brainatlas"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="brainatlas" data.batch_size="4" model="unet" model.num_classes="4" model.num_inp_channels="3" loss="ls" optim="adam" scheduler="step" wandb.project="unet-brainatlas" log_period="50" train.max_epoch="100"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="brainatlas" data.batch_size="4" model="unet" model.num_classes="4" model.num_inp_channels="3" loss="svls" loss.numclasses="4" optim="adam" scheduler="step" wandb.project="unet-brainatlas" log_period="50" train.max_epoch="100"
CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="brainatlas" data.batch_size="4" model="unet" model.num_classes="4" model.num_inp_channels="3" loss="logit_margin" optim="adam" scheduler="step" wandb.project="unet-brainatlas" log_period="50" train.max_epoch="100" 
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="brainatlas" data.batch_size="4" model="unet" model.num_classes="4" model.num_inp_channels="3" loss="focal" loss.gamma="1" optim="adam" scheduler="step" wandb.project="unet-brainatlas"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="brainatlas" data.batch_size="4" model="unet" model.num_classes="4" model.num_inp_channels="3" loss="focal" loss.gamma="2" optim="adam" scheduler="step" wandb.project="unet-brainatlas"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="brainatlas" data.batch_size="4" model="unet" model.num_classes="4" model.num_inp_channels="3" loss="penalty_ent" loss.alpha="0.2" optim="adam" scheduler="step" wandb.project="unet-brainatlas"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="brainatlas" data.batch_size="4" model="unet" model.num_classes="4" model.num_inp_channels="3" loss="penalty_ent" loss.alpha="0.3" optim="adam" scheduler="step" wandb.project="unet-brainatlas"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="brainatlas" data.batch_size="4" model="unet" model.num_classes="4" model.num_inp_channels="3" loss="ls" loss.alpha="0.2" optim="adam" scheduler="step" wandb.project="unet-brainatlas" log_period="50" train.max_epoch="100"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="brainatlas" data.batch_size="4" model="unet" model.num_classes="4" model.num_inp_channels="3" loss="ls" loss.alpha="0.3" optim="adam" scheduler="step" wandb.project="unet-brainatlas" log_period="50" train.max_epoch="100"
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="brainatlas" data.batch_size="4" model="unet" model.num_classes="4" model.num_inp_channels="3" loss="logit_margin" loss.margin="8" optim="adam" scheduler="step" wandb.project="unet-brainatlas" log_period="50" train.max_epoch="100" 
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="brainatlas" data.batch_size="4" model="unet" model.num_classes="4" model.num_inp_channels="3" loss="logit_margin" loss.margin="3" optim="adam" scheduler="step" wandb.project="unet-brainatlas" log_period="50" train.max_epoch="100" 
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="brainatlas" data.batch_size="4" model="unet" model.num_classes="4" model.num_inp_channels="3" loss="logit_margin" loss.margin="0" optim="adam" scheduler="step" wandb.project="unet-brainatlas" log_period="50" train.max_epoch="100" 
CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="brainatlas" data.batch_size="4" model="unet" model.num_classes="4" model.num_inp_channels="3" loss="logit_margin" loss.margin="5" optim="adam" scheduler="step" wandb.project="unet-brainatlas" log_period="50" train.max_epoch="100" 


##unet++
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="brainatlas" data.batch_size="4" model="unetpp" model.num_classes="4" model.num_inp_channels="3" loss="logit_margin" optim="adam" scheduler="step" wandb.project="unet-brainatlas" log_period="50" train.max_epoch="100" 

##unettr
#CUDA_VISIBLE_DEVICES=4 python tools/train_net.py wandb.enable=True task="medseg" data="brainatlas" data.batch_size="4" model="unettr" model.num_classes="4" model.num_inp_channels="3" loss="logit_margin" optim="adam" scheduler="step" wandb.project="unet-brainatlas" log_period="50" train.max_epoch="100" 