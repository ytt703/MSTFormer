if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

for model_name in Informer FEDformer
do
#for seq_len in 96 192 336 720
for seq_len in 720
do
#for pred_len in 96
for pred_len in 192 336 720
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
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 10 \
    --dec_in 10 \
    --c_out 10 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 16 --learning_rate 0.001  | tee logs/LongForecasting/$model_name'_gjemnessund_'$seq_len'_'$pred_len.log

done
done
done


