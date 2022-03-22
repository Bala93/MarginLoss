python tools/train_net.py wandb.enable=False task="medseg" data="prostate" model="unet" loss="ce" optim="adam" scheduler="step" wandb.project="unet-prostate"
