CUDA_VISIBLE_DEVICES=0
python3 main.py \
    --dataset multimodal_few_shot \
    --common-img-side-len 163 \
    --multimodal_few_shot esc50 cifar \
    --num-batches 60000 \
    --model-type gated_conv_1d \
    --embedding-type ConvGRU1d \
    --num-workers 16 \
    --verbose \
    --output-folder real_multimodal_2mode_5w1s\
    # --eval \
    # --output-folder mmaml_5mode_5w1s \
    # --checkpoint tmp/train_dir_0412/mmaml_5mode_5w1s/maml_gated_conv_1d_6000.pt \
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