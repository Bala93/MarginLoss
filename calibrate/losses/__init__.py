from .label_smoothing import LabelSmoothingCrossEntropy
from .focal_loss import FocalLoss

from .penalty_entropy import PenaltyEntropy
#from .penalty_l1 import PenaltyL1
from .logit_margin_l1 import LogitMarginL1
from .ce_dice import CEDiceLoss
from .spatial_ls_repo import CELossWithSVLS_2D
from .logit_margin_l2 import LogitMarginL2
from .logit_margin_ce_dice import LogitMarginDICEL1
from .spatial_adaptive_margin import AdaptMarginSVLS
