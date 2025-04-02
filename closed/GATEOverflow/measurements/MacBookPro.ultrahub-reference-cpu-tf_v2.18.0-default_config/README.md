| Model    | Scenario     |   Accuracy |   Throughput | Latency (in ms)   | Power Efficiency (in samples/J)   | TEST01   | TEST04   |
|----------|--------------|------------|--------------|-------------------|-----------------------------------|----------|----------|
| resnet50 | multistream  |     76.456 |       45.603 | 175.427           |                                   | passed   | passed   |
| resnet50 | singlestream |     76.456 |       20.985 | 47.652            |                                   | passed   | passed   |
| resnet50 | offline      |     76.456 |       67.071 | -                 |                                   | passed   | passed   |