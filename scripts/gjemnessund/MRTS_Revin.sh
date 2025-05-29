if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
#seq_len=336
pred_len=96


for model_name in MRTS_Revin
do
for seq_len in 96 192 336 720
do

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path gjemnessund.h5 \
  --model_id gjemnessund_$seq_len'_'$pred_len \
  --model $model_name \
  --data gjemnessund \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 10 \
  --c_out 10 \
  --e_layers 1 \
  --n_heads 4 \
  --d_model 16 \
  --d_ff 128 \
  --dropout 0.3 \
  --fc_dropout 0.3 \
  --head_dropout 0 \
  --des 'Exp' \
  --itr 1  --batch_size 16 --learning_rate 0.001 |tee logs/LongForecasting/$model_name'_'gjemnessund423_$seq_len'_'$pred_len.log


done
done


