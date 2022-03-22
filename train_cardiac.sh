python tools/train_net.py wandb.enable=True task="medseg" data="cardiac" model="unet" model.num_classes="4" loss="ce" optim="adam" scheduler="step" wandb.project="unet-cardiac"
