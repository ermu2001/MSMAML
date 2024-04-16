
device=cuda:1
output_folder=cifar_5w1s
model_name=maml_gated_conv_1d_6000.pt
checkpoint=train_dir/${output_folder}/${model_name}
python3 main.py \
    --dataset multimodal_few_shot \
    --multimodal_few_shot cifar \
    --common-img-side-len 32 \
    --common-img-channel 3 \
    --num-batches 7500 \
    --eval \
    --meta-batch-size 5 \
    --model-type gated_conv_1d \
    --embedding-type ConvGRU1d \
    --num-workers 16 \
    --verbose \
    --num-classes-per-batch 5 \
    --num-samples-per-class 1 \
    --device ${device} \
    --checkpoint ${checkpoint} \
    --output-folder ${output_folder} > train_dir/${output_folder}/eval--${model_name}.log



output_folder=cifar_5w5s
checkpoint=train_dir/${output_folder}/maml_gated_conv_1d_6000.pt
python3 main.py \
    --dataset multimodal_few_shot \
    --multimodal_few_shot cifar \
    --common-img-side-len 32 \
    --common-img-channel 3 \
    --num-batches 7500 \
    --eval \
    --meta-batch-size 5 \
    --model-type gated_conv_1d \
    --embedding-type ConvGRU1d \
    --num-workers 16 \
    --verbose \
    --num-classes-per-batch 5 \
    --num-samples-per-class 5 \
    --device ${device} \
    --checkpoint ${checkpoint} \
    --output-folder ${output_folder} > train_dir/${output_folder}/eval--${model_name}.log




output_folder=cifar_10w1s
checkpoint=train_dir/${output_folder}/maml_gated_conv_1d_6000.pt
python3 main.py \
    --dataset multimodal_few_shot \
    --multimodal_few_shot cifar \
    --common-img-side-len 32 \
    --common-img-channel 3 \
    --num-batches 7500 \
    --eval \
    --meta-batch-size 5 \
    --model-type gated_conv_1d \
    --embedding-type ConvGRU1d \
    --num-workers 16 \
    --verbose \
    --num-classes-per-batch 10 \
    --num-samples-per-class 1 \
    --device ${device} \
    --checkpoint ${checkpoint} \
    --output-folder ${output_folder} > train_dir/${output_folder}/eval--${model_name}.log



output_folder=cifar_10w5s
checkpoint=train_dir/${output_folder}/maml_gated_conv_1d_6000.pt
python3 main.py \
    --dataset multimodal_few_shot \
    --multimodal_few_shot cifar \
    --common-img-side-len 32 \
    --common-img-channel 3 \
    --num-batches 7500 \
    --eval \
    --meta-batch-size 5 \
    --model-type gated_conv_1d \
    --embedding-type ConvGRU1d \
    --num-workers 16 \
    --verbose \
    --num-classes-per-batch 10 \
    --num-samples-per-class 5 \
    --device ${device} \
    --checkpoint ${checkpoint} \
    --output-folder ${output_folder} > train_dir/${output_folder}/eval--${model_name}.log