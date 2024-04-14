output_folder=brown_first_test

mkdir train_dir/${output_folder}
echo Running Saving to train_dir/${output_folder}
python3 main.py \
    --dataset multimodal_few_shot \
    --multimodal_few_shot brown \
    --common-img-side-len 32 \
    --common-img-channel 3 \
    --num-batches 30000 \
    --meta-batch-size 5 \
    --model-type gated_conv_1d \
    --embedding-type ConvGRU1d \
    --num-workers 16 \
    --verbose \
    --num-classes-per-batch 5 \
    --num-samples-per-class 1 \
    --device cuda:0 \
    --output-folder ${output_folder} > train_dir/${output_folder}/train.log
    # --eval \
    # --output-folder mmaml_5mode_5w1s \
    # --checkpoint tmp/train_dir_0412/mmaml_5mode_5w1s/maml_gated_conv_1d_6000.pt \
    # --mmaml-model True \
