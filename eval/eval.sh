####### Time Perf #######

# usyd baseline
nohup python evaluate.py  \
--config=../config/config_usyd.txt \
--model_config=/home/xiayan/testdir/MinkLoc3D-SI/config/model_config.txt  \
--weights=/home/xiayan/testdir/MinkLoc3D-SI/weights/model_MinkFPNGeM_20220812_101152/epoch40.pth \
> time_usyd_baseline_eval.log 2>&1 & 

# usyd best self_3_cross
nohup python evaluate.py \
--config=../config/config_usyd.txt \
--model_config=/home/xiayan/testdir/MinkLoc3D-SI/config/model_config_self_cross_3.txt \
--weights=/home/xiayan/testdir/MinkLoc3D-SI/weights/model_MinkFPNGeM_20220818_141758/epoch40.pth \
> time_usyd_baseline_eval.log 2>&1

#########################


####### eval kitti #######

# oxford self_3_cross
nohup python evaluate_kitti.py  \
--config=../config/config_kitti.txt \
--model_config=/home/xiayan/testdir/MinkLoc3D-SI/config/model_config_self_cross_3_oxford.txt  \
--weights=/home/xiayan/testdir/MinkLoc3D-SI/weights/model_MinkFPNGeM_20220822_225303/epoch40.pth \
> eval_kitti_oxford_self_cross_3.log 2>&1 &


###########################