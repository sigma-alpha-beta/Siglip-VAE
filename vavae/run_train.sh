config_path=configs/f16d32_siglipv2_long.yaml
WORLD_SIZE=1
RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=7362


torchrun --nproc_per_node=8 \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    main.py \
    --base $config_path \
    --train
