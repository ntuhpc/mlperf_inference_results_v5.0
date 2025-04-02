| Model                                               | Scenario   |   Accuracy |   Throughput | Latency (in ms)   |
|-----------------------------------------------------|------------|------------|--------------|-------------------|
| mobilebert-base_quant-none-bert-99                  | offline    |    90.8056 |       25.813 | -                 |
| mobilebert-14layer_pruned50-none-vnni-bert-99       | offline    |    90.4308 |       60.271 | -                 |
| bert-base-pruned95_obs_quant-none-bert-99           | offline    |    87.8857 |       56.772 | -                 |
| bert-large-pruned80_quant-none-vnni-bert-99         | offline    |    90.2917 |        9.949 | -                 |
| mobilebert-14layer_pruned50_quant-none-vnni-bert-99 | offline    |    90.3672 |      105.283 | -                 |