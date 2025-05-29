if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
#seq_len=720
pred_len=96


for model_name in modernTCN
do
#for pred_len in 96 192 336 720
for seq_len in 96 192 336 720
do

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path gjemnessund.h5 \
  --model_id gjemnessund \
  --model $model_name \
  --data gjemnessund \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --ffn_ratio 8 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 51 \
  --small_size 5 \
  --dims 64 64 64 64 \
  --head_dropout 0.0 \
  --enc_in 10 \
  --dropout 0.9 \
  --des 'Exp' \
  --use_multi_scale False \
  --small_kernel_merged False\
  --itr 1  --batch_size 16 --learning_rate 0.001 |tee logs/LongForecasting/$model_name'_'gjemnessund423_$seq_len'_'$pred_len.log


done
done



