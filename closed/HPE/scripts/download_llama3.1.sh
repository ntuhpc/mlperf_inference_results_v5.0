#reference: https://github.com/mlcommons/inference/tree/master/language/llama3.1-405b

export CHECKPOINT_PATH=models/Llama3/Meta-Llama-3.1-405B-Instruct
git lfs install
git clone https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct ${CHECKPOINT_PATH}
cd ${CHECKPOINT_PATH} && git checkout be673f326cab4cd22ccfef76109faf68e41aa5f1


##preprocessed dataset...
sudo -v ; curl https://rclone.org/install.sh | sudo bash
rclone config create mlc-inference s3 provider=Cloudflare access_key_id=f65ba5eef400db161ea49967de89f47b secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
rclone copy mlc-inference:mlcommons-inference-wg-public/llama3.1_405b/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl ./preprocessed_data/Llama3.1/ -P
