| Model    | Scenario     |   Accuracy |   Throughput | Latency (in ms)   | Power Efficiency (in samples/J)   | TEST01   | TEST04   |
|----------|--------------|------------|--------------|-------------------|-----------------------------------|----------|----------|
| resnet50 | multistream  |     76.456 |       11.36  | 704.255           |                                   | passed   | passed   |
| resnet50 | offline      |     76.456 |       18.379 | -                 |                                   | passed   | passed   |
| resnet50 | singlestream |     76.456 |       16.191 | 61.764            |                                   | passed   | passed   |