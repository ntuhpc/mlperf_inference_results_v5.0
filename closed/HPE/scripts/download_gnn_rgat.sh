#reference from: https://github.com/mlcommons/inference/tree/master/graph/R-GAT
sudo -v ; curl https://rclone.org/install.sh | sudo bash
rclone config create mlc-inference s3 provider=Cloudflare access_key_id=f65ba5eef400db161ea49967de89f47b secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com

rclone copy mlc-inference:mlcommons-inference-wg-public/R-GAT/RGAT.pt $MLPERF_SCRATCH_PATH/models/rgat -P


