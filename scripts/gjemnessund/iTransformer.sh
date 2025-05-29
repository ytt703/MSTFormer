if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi


#seq_len=720
pred_len=96
for model_name in iTransformer
do
for seq_len in 96 192 336 720
#for pred_len in 96 192 336 720
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
    --e_layers 3 \
    --enc_in 10 \
    --dec_in 10 \
    --c_out 10 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 16 \
    --itr 1 --learning_rate 0.001 |tee logs/LongForecasting/$model_name'_'gjemnessund423_$seq_len'_'$pred_len.log

done
done