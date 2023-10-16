# DeepLabv3plus-ILM-VP

# training command:
```
python main.py --dataset cityscapes --data_root datasets/data/cityscapes --model deeplabv3plus_mobilenet --ckpt best_deeplabv3plus_mobilenet_voc_os16.pth --total_itrs 60000 --lr 0.01 --crop_size 513 --batch_size 2 --val_batch_size 4 --output_stride 16 --gpu_id 0
```


# predict command:
```
python predict.py --input datasets/data/cityscapes/leftImg8bit/train/bremen --save_val_results_to test_results
```
