CUDA_VISIBLE_DEVICES=0
python3 main.py \
    --dataset esc50 \
    --num-batches 6000 \
    --output-folder mmaml_5mode_5w1s \
    --model-type gated_conv_1d \
    --embedding-type ConvGRU1d \
    --num-workers 0 \
    --eval \
    --verbose \
    --checkpoint train_dir/mmaml_5mode_5w1s/maml_gated_conv_1d_6000.pt \
    # --mmaml-model True \


# CUDA_VISIBLE_DEVICES=1
# python3 main.py \
#     --dataset esc50 \
#     --num-batches 600000 \
#     --output-folder mmaml_5mode_5w1s \
#     --verbose true\
#     --model-type gatedconv \
#     --embedding-type ConvGRU \
#     --num-workers 0 \
#     # --mmaml-model True \