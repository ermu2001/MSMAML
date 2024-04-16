
device=cuda:0
output_folder=cifar1d-esc50_2m5w1s
mkdir train_dir/${output_folder}
python3 main.py \
    --dataset multimodal_few_shot \
    --multimodal_few_shot cifar1d esc50 \
    --common-img-side-len 163 \
    --common-img-channel 3 \
    --num-batches 30000 \
    --meta-batch-size 5 \
    --model-type gated_conv_1d_noembed \
    --embedding-type ConvGRU1dNoEmbed \
    --num-workers 16 \
    --verbose \
    --num-classes-per-batch 5 \
    --num-samples-per-class 1 \
    --device ${device} \
    --output-folder ${output_folder} > train_dir/${output_folder}/train.log



output_folder=cifar1d-esc50_2m5w5s
mkdir train_dir/${output_folder}
python3 main.py \
    --dataset multimodal_few_shot \
    --multimodal_few_shot cifar1d esc50 \
    --common-img-side-len 32 \
    --common-img-channel 3 \
    --num-batches 30000 \
    --meta-batch-size 5 \
    --model-type gated_conv_1d_noembed \
    --embedding-type ConvGRU1dNoEmbed \
    --num-workers 16 \
    --verbose \
    --num-classes-per-batch 5 \
    --num-samples-per-class 5 \
    --device ${device} \
    --output-folder ${output_folder} > train_dir/${output_folder}/train.log



output_folder=cifar1d-esc50_2m10w1s
mkdir train_dir/${output_folder}
python3 main.py \
    --dataset multimodal_few_shot \
    --multimodal_few_shot cifar1d esc50 \
    --common-img-side-len 32 \
    --common-img-channel 3 \
    --num-batches 30000 \
    --meta-batch-size 5 \
    --model-type gated_conv_1d_noembed \
    --embedding-type ConvGRU1dNoEmbed \
    --num-workers 16 \
    --verbose \
    --num-classes-per-batch 10 \
    --num-samples-per-class 1 \
    --device ${device} \
    --output-folder ${output_folder} > train_dir/${output_folder}/train.log


output_folder=cifar1d-esc50_2m10w5s
mkdir train_dir/${output_folder}
python3 main.py \
    --dataset multimodal_few_shot \
    --multimodal_few_shot cifar1d esc50 \
    --common-img-side-len 32 \
    --common-img-channel 3 \
    --num-batches 30000 \
    --meta-batch-size 5 \
    --model-type gated_conv_1d_noembed \
    --embedding-type ConvGRU1dNoEmbed \
    --num-workers 16 \
    --verbose \
    --num-classes-per-batch 10 \
    --num-samples-per-class 5 \
    --device ${device} \
    --output-folder ${output_folder} > train_dir/${output_folder}/train.log