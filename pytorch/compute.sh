#!/bin/bash
if [ ! -z "$NVPROF" ]
then
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_DIR/lib64
  NVPROF+=" -o keras-${PMIX_RANK}.nvprof"
fi

#python hdf5_to_tfrecord.py --train_dir=$TRAIN_DIR

#python resnet50.py

mkdir /mnt/bb/$USER/log_dir
export STORE_LOG=/mnt/bb/$USER/log_dir

python pytorch_experiment.py --train_dir=$TRAIN_DIR --val_dir=$VAL_DIR --log_dir=$STORE_LOG --batch_size=128

#$NVPROF python -u distributed_pytorch.py --train_dir=$TRAIN_DIR --val_dir=$VAL_DIR --log_dir=$STORE_LOG --batch_size=128 --epochs=90

if [ $PMIX_RANK -eq 0 ]
then
    cp -r /mnt/bb/$USER/log_dir $LOG_DIR 
fi

