sudo -v ; curl https://rclone.org/install.sh | sudo bash

rclone config create mlc-waymo drive config_is_local=false scope=drive.readonly root_folder_id=1xbfnaUurFeXliFFl1i1gj48eRU2NDiH5
#requires auth key
rclone config reconnect mlc-waymo:

rclone copy mlc-waymo:waymo_preprocessed_dataset $MLPERF_SCRATCH_PATH/preprocessed_data/waymo_preprocessed_dataset -P
