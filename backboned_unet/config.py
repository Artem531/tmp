from easydict import EasyDict
import segmentation_models_pytorch

Config = EasyDict()
Config.CLASSES_NAME = ('Empty', 'EAN', 'QRCode', 'Postnet',
                        'DataMatrix', 'PDF417', 'Aztec')

Config.head_channel = 64

# train
Config.lr = 1e-3
Config.AMSGRAD = True
Config.max_iter = 1000000
Config.lr_schedule = 'WarmupMultiStepLR'
Config.gamma = 0.1
Config.steps = (35000, 40000)
Config.warmup_iters = 1000

# backbone
Config.slug = 'resnet18'
Config.freeze_backbone = True
Config.backbone_checkpoint_dir = '/home/artem/PycharmProjects/CenterNetRefs/centerNetWithSelfSupervisedFeatureLearning/backbone/testBackBone'
Config.loadBackBone = False

# dataset
Config.num_classes = len(Config.CLASSES_NAME) #2
Config.batch_size = 4 # 128
Config.root =  '/media/artem/A2F4DEB0F4DE85C7/myData/datasets/barcodes/ZVZ-real-512/'
Config.resize_size = [512, 512]
Config.num_workers = 1
Config.mean = [0.40789654, 0.44719302, 0.47026115]
Config.std = [0.28863828, 0.27408164, 0.27809835]

# loss
mode = 'multiclass'
Config.loss_f = segmentation_models_pytorch.losses.FocalLoss(mode)

# other
Config.eval = True
Config.gpu = True
Config.resume = False

Config.init = False
Config.init_checkpoint_dir = "/home/artem/PycharmProjects/CenterNetRefs/simple-centernet-pytorch/ckp/best_checkpoint.pth"

Config.log_dir = './log'
Config.checkpoint_dir = './ckp'

Config.log_interval = 20
Config.apex = False

Config.loss_alpha = 1.
Config.loss_beta = 0.1
Config.loss_gamma = 1.
Config.down_stride = 4

Config.f1Beta = 2
