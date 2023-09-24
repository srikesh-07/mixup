GPU=0
DATA=("DD" "PROTEINS" "PTC" "IMDBBINARY" "FRANKENSTEIN")

for dataset in "${DATA[@]}";
do
  CUDA_VISIBLE_DEVICES=${GPU} python mixup_v2.py --dataset $dataset --batch_size 32
done

