# DeepLabv3plus-ILM-VP

# training command:
```
python main.py --model deeplabv3plus_mobilenet --dataset cityscapes --gpu_id 0,1  --lr 0.1  --crop_size 513 --batch_size 16 --output_stride 16 --data_root ./datasets/data/cityscapes
```



# testing command:
```
python predict.py --input datasets/data/cityscapes/leftImg8bit/train/bremen --save_val_results_to test_results
```
