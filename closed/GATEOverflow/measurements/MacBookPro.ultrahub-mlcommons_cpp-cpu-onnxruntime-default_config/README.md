| Model    | Scenario     |   Accuracy |   Throughput | Latency (in ms)   | Power Efficiency (in samples/J)   | TEST01   | TEST04   |
|----------|--------------|------------|--------------|-------------------|-----------------------------------|----------|----------|
| resnet50 | multistream  |     76.456 |       34.752 | 230.202           |                                   | passed   | passed   |
| resnet50 | singlestream |     76.456 |       27.402 | 36.494            |                                   | passed   | passed   |
| resnet50 | offline      |     76.456 |       38.76  | -                 |                                   | passed   | passed   |