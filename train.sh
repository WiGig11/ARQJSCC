#python train.py --ckpt_addr '' --batch_size 64 --device 1 --max_epoches 500 --check_val_every_n_epoch 50 --save_ckpt_every_n_epochs 100 --fast_dev_run False --channel 'Awgn'
#python train_jscc.py --ckpt_addr '' --batch_size 64 --device 1 --max_epoches 1000 --check_val_every_n_epoch 50 --save_ckpt_every_n_epochs 100 --fast_dev_run False --channel 'Rayl'
python train_jscc.py --ckpt_addr '' --in_channels 8 --out_channels 8 --batch_size 64 --device [0] --max_epoches 10 --check_val_every_n_epoch 5 --save_ckpt_every_n_epochs 5 --fast_dev_run False --channel 'Awgn'
#awgn deepjscc 1/12
#python train_jscc.py --ckpt_addr '' --in_channels 16 --out_channels 16 --batch_size 64 --device 2 --max_epoches 1000 --check_val_every_n_epoch 50 --save_ckpt_every_n_epochs 100 --fast_dev_run True --channel 'Awgn'
#awgn deepjscc 1/6
