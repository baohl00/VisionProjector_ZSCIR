DATASET=circo # fiq-shirt, fiq-dress, fiq-toptee, circo, cirr 
SIZE=large
k=51
NOTE=resmlp # vision projector type 

echo "-- INFERENCE of DATA: ${DATASET} --"

python inference.py \
	--model_size $SIZE \
	--model_path ./models/magic_lens_clip_$SIZE.pkl \
	--dataset $DATASET \
	--top_k $k \
	--note $NOTE 

echo ">>> Evaluation: dataset $DATASET , size ${SIZE}, top_${k}, note ${NOTE}"
