CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="abdomen" loss="ce" model="unet" model.num_inp_channels="1" model.num_classes="5" test.checkpoint="/home/ar88770/MarginLoss/outputs/abdomen/unet-ce-adam/20220619-21:08:32-970895/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/abdomen/unet-ce-adam/20220619-21:08:32-970895" wandb.enable=True wandb.project="unet-abdomen-test"