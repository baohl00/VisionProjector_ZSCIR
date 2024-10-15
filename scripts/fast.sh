DATASET=cirr # fiq-shirt, fiq-dress, fiq-toptee, circo, cirr

declare -a SIZE=("base" "large")

k=51
exp=16
n=4
NOTE=resmlp${n}x_expansion$exp # attn features = dim//4

echo "-- INFERENCE of DATA: ${DATASET} --"

for size_i in "${SIZE[@]}"
do
	python inference.py \
		--model_size $size_i \
		--model_path ./models/magic_lens_clip_$size_i.pkl \
		--dataset $DATASET \
		--top_k $k \
		--note $NOTE # iat = img and txt
			
	echo ">>> Evaluation: dataset $DATASET , size ${size_i}, top_${k}, note ${NOTE}"
done
#python3 eval.py \
#	--predict ./output/circo_${SIZE}_${k}gmlp/circo_results.json  
