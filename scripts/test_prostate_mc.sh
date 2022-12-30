#CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="4" loss="ce" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ce-adam/20220618-19:45:47-605216/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ce-adam/20220618-19:45:47-605216" wandb.enable=True wandb.project="unet-prostate_mc-test"
#CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="16" loss="ce_dice" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ce_dice-adam/20220618-19:59:27-919203/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ce_dice-adam/20220618-19:59:27-919203" wandb.enable=True wandb.project="unet-prostate_mc-test"
#CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="16" loss="focal" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-focal-adam/20220619-19:27:36-214693/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-focal-adam/20220619-19:27:36-214693" wandb.enable=True wandb.project="unet-prostate_mc-test"
#CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="16" loss="penalty_ent" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-penalty_ent-adam/20220618-20:26:38-657056/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-penalty_ent-adam/20220618-20:26:38-657056" wandb.enable=True wandb.project="unet-prostate_mc-test"
#CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="16" loss="ls" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ls-adam/20220619-18:30:02-915430/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ls-adam/20220619-18:30:02-915430" wandb.enable=True wandb.project="unet-prostate_mc-test"
#CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="16" loss="svls" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-svls-adam/20220618-20:55:27-830091/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-svls-adam/20220618-20:55:27-830091" wandb.enable=True wandb.project="unet-prostate_mc-test"
#UDA_VISIBLE_DEVICES=5 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="4" loss="logit_margin" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-logit_margin-adam/20220618-21:09:31-193817/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-logit_margin-adam/20220618-21:09:31-193817" wandb.enable=True wandb.project="unet-prostate_mc-test"


## temperature 
ts_dir="/home/ar88770/MarginLoss/misc/calibration_Tiramisu"
CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="4" loss="ce" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ce-adam/20220618-19:45:47-605216/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ce-adam/20220618-19:45:47-605216" wandb.enable=True wandb.project="unet-prostate_mc-test" test.post_temperature=True test.ts_type="ts" test.ts_path="${ts_dir}/TS_ce_promise_mc_model_best.pth.tar"
CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="4" loss="ce_dice" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ce_dice-adam/20220618-19:59:27-919203/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ce_dice-adam/20220618-19:59:27-919203/" wandb.enable=True wandb.project="unet-prostate_mc-test" test.post_temperature=True test.ts_type="ts" test.ts_path="${ts_dir}/TS_ce_dice_promise_mc_model_best.pth.tar"
CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="4" loss="focal" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-focal-adam/20220619-19:27:36-214693/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-focal-adam/20220619-19:27:36-214693" wandb.enable=True wandb.project="unet-prostate_mc-test" test.post_temperature=True test.ts_type="ts" test.ts_path="${ts_dir}/TS_focal_promise_mc_model_best.pth.tar"
CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="4" loss="penalty_ent" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-penalty_ent-adam/20220618-20:26:38-657056/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-penalty_ent-adam/20220618-20:26:38-657056/" wandb.enable=True wandb.project="unet-prostate_mc-test" test.post_temperature=True test.ts_type="ts" test.ts_path="${ts_dir}/TS_penalty_promise_mc_model_best.pth.tar" 
CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="4" loss="ls" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ls-adam/20220619-18:30:02-915430/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ls-adam/20220619-18:30:02-915430" wandb.enable=True wandb.project="unet-prostate_mc-test" test.post_temperature=True test.ts_type="ts" test.ts_path="${ts_dir}/TS_ls_promise_mc_model_best.pth.tar"
CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="4" loss="svls" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-svls-adam/20220618-20:55:27-830091/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-svls-adam/20220618-20:55:27-830091" wandb.enable=True wandb.project="unet-prostate_mc-test" test.post_temperature=True test.ts_type="ts" test.ts_path="${ts_dir}/TS_svls_promise_mc_model_best.pth.tar"
CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="4" loss="logit_margin" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-logit_margin-adam/20220618-21:09:31-193817/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-logit_margin-adam/20220618-21:09:31-193817" wandb.enable=True wandb.project="unet-prostate_mc-test" test.post_temperature=True test.ts_type="ts" test.ts_path="${ts_dir}/TS_margin_promise_mc_model_best.pth.tar" 




## temperature 
#ts_dir="/home/ar88770/MarginLoss/misc/calibration_Tiramisu"
#CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="4" loss="ce" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ce-adam/20220618-19:45:47-605216/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ce-adam/20220409-22:47:25-847975" wandb.enable=True wandb.project="unet-prostate_mc-test" test.post_temperature=True test.ts_type="grid" 
#CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="4" loss="ce_dice" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ce_dice-adam/20220618-19:59:27-919203/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ce_dice-adam/20220618-19:59:27-919203/" wandb.enable=True wandb.project="unet-prostate_mc-test" test.post_temperature=True test.ts_type="grid" 
#CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="4" loss="focal" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-focal-adam/20220619-19:27:36-214693/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-focal-adam/20220619-19:27:36-214693" wandb.enable=True wandb.project="unet-prostate_mc-test" test.post_temperature=True test.ts_type="grid"
#CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="4" loss="penalty_ent" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-penalty_ent-adam/20220618-20:26:38-657056/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-penalty_ent-adam/20220618-20:26:38-657056/" wandb.enable=True wandb.project="unet-prostate_mc-test" test.post_temperature=True test.ts_type="grid"
#CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="4" loss="ls" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ls-adam/20220619-18:30:02-915430/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ls-adam/20220619-18:30:02-915430" wandb.enable=True wandb.project="unet-prostate_mc-test" test.post_temperature=True test.ts_type="grid" 
#CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="4" loss="svls" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-svls-adam/20220618-20:55:27-830091/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-svls-adam/20220618-20:55:27-830091" wandb.enable=True wandb.project="unet-prostate_mc-test" test.post_temperature=True test.ts_type="grid" 
#CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="4" loss="logit_margin" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-logit_margin-adam/20220618-21:09:31-193817/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-logit_margin-adam/20220618-21:09:31-193817" wandb.enable=True wandb.project="unet-prostate_mc-test" test.post_temperature=True test.ts_type="grid" 
#





## dilation
#CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="4" loss="ce" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ce-adam/20220409-22:47:25-847975/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ce-adam/20220409-22:47:25-847975" wandb.enable=True wandb.project="unet-prostate_mc-test" test.is_dilate=True
#CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="4" loss="ce_dice" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ce_dice-adam/20220427-13:38:39-924288/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ce_dice-adam/20220427-13:38:39-924288" wandb.enable=True wandb.project="unet-prostate_mc-test" test.is_dilate=True
#CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="4" loss="focal" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-focal-adam/20220406-17:06:46-112063/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-focal-adam/20220406-17:06:46-112063" wandb.enable=True wandb.project="unet-prostate_mc-test" test.is_dilate=True
#CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="4" loss="penalty_ent" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-penalty_ent-adam/20220411-23:10:20-338365/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-penalty_ent-adam/20220411-23:10:20-338365" wandb.enable=True wandb.project="unet-prostate_mc-test" test.is_dilate=True
#CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="4" loss="ls" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ls-adam/20220409-22:57:05-534513/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ls-adam/20220409-22:57:05-534513" wandb.enable=True wandb.project="unet-prostate_mc-test" test.is_dilate=True
#CUDA_VISIBLE_DEVICES=4 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="4" loss="svls" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-svls-adam/20220417-14:10:04-419455/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-svls-adam/20220417-14:10:04-419455" wandb.enable=True wandb.project="unet-prostate_mc-test" test.is_dilate=True
#CUDA_VISIBLE_DEVICES=5 python tools/test_net.py task="medseg" data="prostate_mc" data.batch_size="4" loss="logit_margin" model="unet" model.num_inp_channels="1" model.num_classes="3" test.checkpoint="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-logit_margin-adam/20220409-23:07:04-480575/best.pth" hydra.run.dir="/home/ar88770/MarginLoss/outputs/prostate_mc/unet-logit_margin-adam/20220409-23:07:04-480575" wandb.enable=True wandb.project="unet-prostate_mc-test" test.is_dilate=True