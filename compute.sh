#python hdf5_to_tfrecord.py --train_dir=$TRAIN_DIR

#python resnet50.py

export STORE_LOG=/mnt/bb/$USER/log_dir

python pytorch_experiment.py --train_dir=$TRAIN_DIR --val_dir=$VAL_DIR --log_dir=$STORE_LOG --verbose=True

if [ $PMIX_RANK -eq 0 ]
then
    cp -r /mnt/bb/$USER/log_dir $LOG_DIR 
fi

