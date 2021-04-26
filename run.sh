#!/bin/bash
#BSUB -J bert_pretrain
#BSUB -q gpu_v100
#BSUB -m "gpu15"
#BSUB -gpu num=6
#BSUB -outdir "./output"
#BSUB -o ./output/%J.out -e ./output/%J.err
python main.py
