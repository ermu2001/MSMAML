
device=cuda:1
model_name=maml_gated_conv_1d_3000.pt
output_folder=brown_5w1s
checkpoint=train_dir/${output_folder}/${model_name}
python3 main.py \
    --dataset multimodal_few_shot \
    --multimodal_few_shot brown \
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


output_folder=brown_5w5s
checkpoint=train_dir/${output_folder}/maml_gated_conv_1d_6000.pt
python3 main.py \
    --dataset multimodal_few_shot \
    --multimodal_few_shot brown \
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




output_folder=brown_10w1s
checkpoint=train_dir/${output_folder}/maml_gated_conv_1d_6000.pt
python3 main.py \
    --dataset multimodal_few_shot \
    --multimodal_few_shot brown \
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



output_folder=brown_10w5s
checkpoint=train_dir/${output_folder}/maml_gated_conv_1d_6000.pt
python3 main.py \
    --dataset multimodal_few_shot \
    --multimodal_few_shot brown \
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
