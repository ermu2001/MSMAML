# CUDA_VISIBLE_DEVICES=0,1,2
python3 main.py \
    --dataset brown \
    --num-batches 30000 \
    --output-folder mmaml_1modet_20w1s \
    --model-type gated_conv_1d \
    --embedding-type ConvGRU1d \
    --num-workers 16 \
    --verbose \
    --num-classes-per-batch 10 \
    --num-samples-per-class 1 \
    --device cuda:0 \
    # --eval \
    # --checkpoint train_dir/mmaml_1modea_5w1s/maml_gated_conv_1d_12000.pt \
    # --mmaml-model True \

python3 main.py \
    --dataset brown \
    --num-batches 30000 \
    --output-folder mmaml_1modet_20w5s \
    --model-type gated_conv_1d \
    --embedding-type ConvGRU1d \
    --num-workers 16 \
    --verbose \
    --num-classes-per-batch 10 \
    --num-samples-per-class 5 \
    --device cuda:0 \