#!/bin/bash
python style.py \
    -s ../data/style/starry_night.jpg \
    -c ../data/content/sanfrancisco.jpg \
    -o ../data/styled/sanfrancisco_starry.jpg \
    -t ../data/tem \
    -i "content" \
    -m 5000 \
    -v 100 \
    -x 1e-4 \
    -y 0.2 \
    -z 1e-2 \
    -r 1e0 \
    -g 1



    
