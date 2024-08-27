#!/bin/bash
cd ../..
DATA=../DATA
TRAINER=Meta_B2N_AUG
SHOTS=16
LOADEP=10
CFG=vit_b16_aug
DATASET=fgvc_aircraft

for SEED in 1 2 3 4 5
do 

    COMMON_DIR=${DATASET}/${TRAINER}/${CFG}_shots_${SHOTS}/seed${SEED}
    DIRTRAIN=./B2N_AUG/train_base/${COMMON_DIR}
    DIRTEST=./B2N_AUG/test_new/${COMMON_DIR}

    if [ -d "$DIRTRAIN" ]; then
        echo "Oops! The results exist at ${DIRTRAIN} (so skip this job)"
    else
        CUDA_VISIBLE_DEVICES=0 python train_aug.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/b2n/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIRTRAIN} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES base \
        TRAINER.META.PREC 'fp16'
    fi

    if [ -d "$DIRTEST" ]; then
        echo "Oops! The results exist at ${DIRTEST} (so skip this job)"
    else
        CUDA_VISIBLE_DEVICES=0 python train_aug.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/b2n/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIRTEST} \
        --model-dir ${DIRTRAIN} \
        --load-epoch ${LOADEP} \
        --eval-only \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES new
    fi

done