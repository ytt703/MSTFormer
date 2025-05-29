if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

#for model_name in Informer
for model_name in Informer FEDformer
do
for seq_len in 720
do
for pred_len in 192 336 720
do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path bergsoysund.h5 \
    --model_id bergsoysund_$pred_len \
    --model $model_name \
    --data bergsoysund \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 23 \
    --dec_in 23 \
    --c_out 23 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 16 --learning_rate 0.001  | tee logs/LongForecasting/$model_name'_bergsoysund424_'$seq_len'_'$pred_len.log

done
done
done


