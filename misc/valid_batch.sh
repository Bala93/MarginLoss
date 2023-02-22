#<<FLARE
DATASET='abdomen'
BASE_DIR='/home/ar88770/MarginLoss/outputs'
METHOD_NAME='unet-penalty_ent-adam'
MODEL_NAME='best.pth'
CKPT='20220620-13:38:47-558393'
python valid3d.py --dataset_type 'flare' --model_path ${BASE_DIR}'/'${DATASET}'/'${METHOD_NAME}'/'${CKPT}'/'${MODEL_NAME} --ece_choice 'fg'
#FLARE

<<ACDC
DATASET='cardiac'
BASE_DIR='/home/ar88770/MarginLoss/outputs'
METHOD_NAME='unet-adaptive_margin_svls-adam'
MODEL_NAME='best.pth'
CKPT='20230221-15:54:04-690204'
python valid3d.py --dataset_type 'acdc' --model_path ${BASE_DIR}'/'${DATASET}'/'${METHOD_NAME}'/'${CKPT}'/'${MODEL_NAME} --ece_choice 'fgbnd'
ACDC


<<BraTS
DATASET='brain19'
BASE_DIR='/home/ar88770/MarginLoss/outputs'
METHOD_NAME='unet-adaptive_margin_svls-adam'
MODEL_NAME='best.pth'
CKPT='20230221-08:42:09-561753'
python valid3d.py --dataset_type 'brats19' --model_path ${BASE_DIR}'/'${DATASET}'/'${METHOD_NAME}'/'${CKPT}'/'${MODEL_NAME} --ece_choice 'fg'
BraTS

