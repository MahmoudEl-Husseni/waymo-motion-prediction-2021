import os

class DIR: 
    MAIN_DIR = "/main/Datasets/waymo/testing"

    DATA_DIR = os.path.join(MAIN_DIR, "render")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "val")
    TEST_DIR = os.path.join(DATA_DIR, "test")

    OUT_DIR = os.path.join(MAIN_DIR, "cnn_impl_output")
    TB_DIR = os.path.join(OUT_DIR, "tb")
    VIS_DIR = os.path.join(OUT_DIR, "vis")
    CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")

MODEL_NAME = "xception71"
IN_CHANNELS = 25
N_TRAJ = 6
FUTURE_TS = 80

TRAIN_BS = 4
VAL_BS = 8
TEST_BS = 4
N_WORKERS = 1

LOG_STEP = 10

LR = 1e-3
DEVICE = 'cuda'
EPOCHS = 100
LOSS = 'neg_multi_log_likelihood'

VIS_HEIGHT = 640
VIS_WIDTH = 480

LOAD_MODEL = False
LOAD_MODEL_FILE = "last_model.pth"

CKPT_EPOCH = 10