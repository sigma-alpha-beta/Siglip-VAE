CONFIG_PATH=$1

PRECISION=${PRECISION:-bf16}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-12357}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# # export NCCL_P2P_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0


accelerate launch \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --num_processes  $(($GPUS_PER_NODE*$NNODES)) \
    --num_machines $NNODES \
    --mixed_precision $PRECISION \
    train.py \
    --config $CONFIG_PATH
