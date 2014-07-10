#!/bin/bash
source /home/dosovits/.bashrc
export PYTHONPATH="/home/dosovits/Python/HPOlib/benchmarks/imagenet-augment-by-default"
cd /home/dosovits/Python/HPOlib/benchmarks/imagenet-augment-by-default

HPOlib-run -o /home/dosovits/Python/HPOlib/optimizers/smac/smac -s 13


