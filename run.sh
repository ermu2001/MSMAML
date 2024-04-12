# CUDA_VISIBLE_DEVICES=0
# python3 main.py \
#     --dataset cifar \
#     --num-batches 600000 \
#     --output-folder mmaml_5mode_5w1s \
#     --verbose true\
#     --model-type gated_conv_1d \
#     --embedding-type ConvGRU1d \
#     # --mmaml-model True \


CUDA_VISIBLE_DEVICES=1
python3 main.py \
    --dataset cifar \
    --num-batches 600000 \
    --output-folder mmaml_5mode_5w1s \
    --verbose true\
    --model-type gatedconv \
    --embedding-type ConvGRU \
    # --mmaml-model True \