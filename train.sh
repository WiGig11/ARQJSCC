#python train.py --ckpt_addr '' --batch_size 64 --device 1 --max_epoches 500 --check_val_every_n_epoch 50 --save_ckpt_every_n_epochs 100 --fast_dev_run False --channel 'Awgn'
python train_jscc.py --ckpt_addr '' --batch_size 64 --device 1 --max_epoches 1000 --check_val_every_n_epoch 50 --save_ckpt_every_n_epochs 100 --fast_dev_run False --channel 'Rayl'