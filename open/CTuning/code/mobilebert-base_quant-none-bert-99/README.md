# Testing MLCommons CMX with MLPerf automations

This open submission is intended to test a prototype of the new 
[CMX automation framework](https://github.com/mlcommons/ck/tree/master/cmx)
with [MLPerf automations](https://access.cknowledge.org/playground/?action=scripts).

## Install on Ubuntu

See the [CMX installation guide](https://access.cknowledge.org/playground/?action=install).

```bash
pip install -U cmind
cmx test core

pip install -U cmx4mlperf

cmlc pull repo mlcommons@mlperf-automations
```

## Find performance

```bash
cr run-mlperf,inference,_find-performance,_full,_r5.0-dev \
   --model=bert-99 \
   --implementation=reference \
   --framework=deepsparse \
   --category=datacenter \
   --scenario=Offline \
   --execution_mode=test \
   --device=cpu  \
   --quiet \
   --test_query_count=100 \
   --nm_model_zoo_stub=zoo:nlp/question_answering/mobilebert-none/pytorch/huggingface/squad/base_quant-none
```

## Run benchmark in Offline mode

```bash
cr run-mlperf,inference,_full,_r5.0-dev \
   --model=bert-99 \
   --implementation=reference \
   --framework=deepsparse \
   --category=datacenter \
   --scenario=Offline \
   --execution_mode=valid \
   --device=cpu \
   --target_qps=54 \
   --quiet \
   --nm_model_zoo_stub=zoo:nlp/question_answering/mobilebert-none/pytorch/huggingface/squad/base_quant-none
```

## Future work

Learn more about our community initiatives to co-design more efficient and cost-effective AI/ML systems 
with the support of MLCommons CM/CMX, virtualized MLOps, MLPerf automations, Collective Knowledge Playground,
and reproducible optimization tournaments in our [white paper](https://arxiv.org/abs/2406.16791).
