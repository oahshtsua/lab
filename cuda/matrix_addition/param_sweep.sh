#!/bin/bash

MATRIX_ROWS=(10 100 1000 500 100)
MATRIX_COLS=(10 100 1000 2000 10000)
BLOCKDIM_X=(16 16 32)
BLOCKDIM_Y=(16 32 16)

mat_n=${#MATRIX_ROWS[@]}
dim_n=${#BLOCKDIM_X[@]}

for ((i=0; i<$dim_n; i++))
do
    for ((j=0; j<$mat_n; j++))
    do
        xdim=${BLOCKDIM_X[$i]}
        ydim=${BLOCKDIM_Y[$i]}
        rows=${MATRIX_ROWS[$j]}
        cols=${MATRIX_COLS[$j]}

        ./matadd -r $rows -c $cols -x $xdim -y $ydim
    done
done
