#conda activate campolina
#srun --pty --ntasks=4 --gres gpu:rtxa4000:1 --qos scavenger --account scavenger --partition scavenger --mem 32G --time 03:00:00 bash

python ../../modification/src/main.py > train.out