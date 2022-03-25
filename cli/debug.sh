#rm -r saved/Debug
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch --nproc_per_node=1 \
                                   --nnodes=1 \
                                   --node_rank=0 \
                                   --master_addr=10.1.38.150 \
                                   --master_port=1557 \
                                   train.py \
                                   --config config/debug.yaml \
                                   --freq-log 1
