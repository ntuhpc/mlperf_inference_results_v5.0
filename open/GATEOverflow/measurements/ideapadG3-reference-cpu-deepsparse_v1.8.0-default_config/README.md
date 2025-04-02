| Model                                         | Scenario     |   Accuracy |   Throughput | Latency (in ms)   |
|-----------------------------------------------|--------------|------------|--------------|-------------------|
| bert-base-pruned95_obs_quant-none-bert-99     | singlestream |    87.8857 |       27.254 | 36.692            |
| bert-base-pruned95_obs_quant-none-bert-99     | offline      |    87.8857 |       46.684 | -                 |
| mobilebert-base_quant-none-bert-99            | singlestream |    90.8127 |       17.8   | 56.181            |
| mobilebert-base_quant-none-bert-99            | offline      |    90.8056 |       19.52  | -                 |
| mobilebert-14layer_pruned50-none-vnni-bert-99 | offline      |    90.4308 |       57.712 | -                 |