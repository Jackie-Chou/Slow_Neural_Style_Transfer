#!/bin/bash
python style.py \
    -s ../data/style/wave.jpg \
    -c ../data/content/000000581781.jpg \
    -o ../data/styled/000000581781_wave.jpg \
    -t ../data/tem \
    -i "content" \
    -m 500 \
    -v 100 \
    -x 1e-3 \
    -y 0.2 \
    -z 1e-5 \
    -r 1e3 \
    -g 1



    
