DATASET=circo # fiq-shirt, fiq-dress, fiq-toptee, circo
SIZE=large
k=51
NOTE=gmlp_concat16_avg # attn features = dim//4

echo "-- INFERENCE of DATA: ${DATASET} --"

python inference.py \
	--model_size $SIZE \
	--model_path ./models/magic_lens_clip_$SIZE.pkl \
	--dataset $DATASET \
	--top_k $k \
	--note $NOTE # iat = img and txt

echo ">>> Evaluation: dataset $DATASET , size ${SIZE}, top_${k}, note ${NOTE}"
#python3 eval.py \
#	--predict ./output/circo_${SIZE}_${k}gmlp/circo_results.json  
