module load anaconda
source activate gemini

./distributed_train.sh 2 /cifs/data/tserre_lrs/projects/projects/prj_video_imagenet/imagenet/ --model efficientnet_b2 -b 128 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.3 --drop-path 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .016


./distributed_train.sh 2 /oscar/data/tserre/data/ImageNet/ILSVRC/Data/CLS-LOC --model seresnet34 --sched cosine --epochs 150 --warmup-epochs 5 --lr 0.4 --reprob 0.5 --remode pixel --batch-size 256 --amp -j 4


./distributed_train.sh 2 /oscar/data/tserre/data/ImageNet/ILSVRC/Data/CLS-LOC --model seresnet34 --sched cosine --epochs 150 --warmup-epochs 5 --lr 0.1 --reprob 0.5 --remode pixel --batch-size 1024 --amp -j 4

