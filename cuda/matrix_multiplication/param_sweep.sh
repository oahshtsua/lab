#!/bin/bash

TILE_SIZES=(2, 4, 8 16 32)

for TILE in "${TILE_SIZES[@]}"; do
    echo "Running ./matmul with tile size = $TILE"
    ./matmul $TILE
done

