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
#for pred_len in 96 192 336 720
for seq_len in 96 192 336 720
do


  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path bergsoysund.h5 \
    --model_id bergsoysund \
    --model $model_name \
    --data bergsoysund \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 3 \
    --enc_in 23 \
    --dec_in 23 \
    --c_out 23 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 16 \
    --itr 1 --learning_rate 0.001 |tee logs/LongForecasting/$model_name'_'bergsoysund424_$seq_len'_'$pred_len.log

done
done