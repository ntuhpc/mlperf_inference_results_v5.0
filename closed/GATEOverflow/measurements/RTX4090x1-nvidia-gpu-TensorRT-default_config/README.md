| Model     | Scenario     |   Accuracy |   Throughput | Latency (in ms)   | Power Efficiency (in samples/J)   | TEST01   |
|-----------|--------------|------------|--------------|-------------------|-----------------------------------|----------|
| retinanet | offline      |     37.344 |      869.415 | -                 |                                   | passed   |
| retinanet | singlestream |     37.338 |      588.235 | 1.7               |                                   | passed   |
| retinanet | server       |     37.283 |      637.535 | -                 |                                   | passed   |
| retinanet | multistream  |     37.328 |      690.846 | 11.58             |                                   | passed   |