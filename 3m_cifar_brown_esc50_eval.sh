output_folder=cifar_brown_esc50_3m5w1s
check_point=maml_gated_conv_1d_6000.pt
mkdir train_dir/${output_folder}
python3 main.py \
    --dataset multimodal_few_shot \
    --multimodal_few_shot cifar brown esc50\
    --common-img-side-len 32 \
    --common-img-channel 3 \
    --num-batches 7500 \
    --meta-batch-size 5 \
    --model-type gated_conv_1d \
    --embedding-type ConvGRU1d \
    --num-workers 16 \
    --verbose \
    --num-classes-per-batch 5 \
    --num-samples-per-class 1 \
    --device cuda:0 \
    --eval \
    --checkpoint train_dir/${output_folder}/${check_point} \
    --output-folder ${output_folder} > train_dir/${output_folder}/eval.log

output_folder=cifar_brown_esc50_3m5w5s
mkdir train_dir/${output_folder}
python3 main.py \
    --dataset multimodal_few_shot \
    --multimodal_few_shot cifar brown esc50\
    --common-img-side-len 32 \
    --common-img-channel 3 \
    --num-batches 7500 \
    --meta-batch-size 5 \
    --model-type gated_conv_1d \
    --embedding-type ConvGRU1d \
    --num-workers 16 \
    --verbose \
    --num-classes-per-batch 5 \
    --num-samples-per-class 5 \
    --device cuda:0 \
    --eval \
    --checkpoint train_dir/${output_folder}/${check_point} \
    --output-folder ${output_folder} > train_dir/${output_folder}/eval.log

output_folder=cifar_brown_esc50_3m10w1s
mkdir train_dir/${output_folder}
python3 main.py \
    --dataset multimodal_few_shot \
    --multimodal_few_shot cifar brown esc50\
    --common-img-side-len 32 \
    --common-img-channel 3 \
    --num-batches 7500 \
    --meta-batch-size 5 \
    --model-type gated_conv_1d \
    --embedding-type ConvGRU1d \
    --num-workers 16 \
    --verbose \
    --num-classes-per-batch 10 \
    --num-samples-per-class 1 \
    --device cuda:0 \
    --eval \
    --checkpoint train_dir/${output_folder}/${check_point} \
    --output-folder ${output_folder} > train_dir/${output_folder}/eval.log

output_folder=cifar_brown_esc50_3m10w5s
mkdir train_dir/${output_folder}
python3 main.py \
    --dataset multimodal_few_shot \
    --multimodal_few_shot cifar brown esc50\
    --common-img-side-len 32 \
    --common-img-channel 3 \
    --num-batches 7500 \
    --meta-batch-size 5 \
    --model-type gated_conv_1d \
    --embedding-type ConvGRU1d \
    --num-workers 16 \
    --verbose \
    --num-classes-per-batch 10 \
    --num-samples-per-class 5 \
    --device cuda:0 \
    --eval \
    --checkpoint train_dir/${output_folder}/${check_point} \
    --output-folder ${output_folder} > train_dir/${output_folder}/eval.log