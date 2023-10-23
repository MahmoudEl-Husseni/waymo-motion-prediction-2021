MODEL_NAME=xception71
python train.py \
    --train-data ./train \
    --dev-data ./dev \
    --save ./${MODEL_NAME} \
    --model ${MODEL_NAME} \
    --img-res 224 \
    --in-channels 25 \
    --time-limit 80 \
    --n-traj 6 \
    --lr 0.001 \
    --batch-size 48 \
    --n-epochs 120
