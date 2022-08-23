cd /home/xiayan/testdir/MinkLoc3D-SI/training

############### Oxford ###############

nohup python train.py \
--config=../config/config_oxford.txt \
--model_config=../config/model_config_self_cross_3_oxford.txt \
> oxford_5m_3ds_baseline.log 2>&1 &

######################################