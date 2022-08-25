cd /home/xiayan/testdir/MinkLoc3D-SI/training

############### Oxford ###############

nohup python train.py \
--config=../config/config_oxford.txt \
--model_config=../config/model_config_self_cross_3_oxford.txt \
> oxford_5m_3ds_cross_self_3.log 2>&1 &

nohup python train.py \
--config=../config/config_oxford.txt \
--model_config=../config/model_config.txt \
> oxford_5m_3ds_baseline.log 2>&1 &

######################################

############### TUM ###############

nohup python train.py \
--config=../config/config_tum.txt \
--model_config=../config/model_config.txt \
> tum_5m_3ds_baseline.log 2>&1 &

######################################