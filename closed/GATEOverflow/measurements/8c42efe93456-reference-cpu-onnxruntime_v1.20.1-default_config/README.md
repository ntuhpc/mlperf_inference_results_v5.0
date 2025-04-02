| Model    | Scenario     |   Accuracy |   Throughput | Latency (in ms)   | Power Efficiency (in samples/J)   | TEST01   | TEST04   |
|----------|--------------|------------|--------------|-------------------|-----------------------------------|----------|----------|
| resnet50 | multistream  |     76.456 |       16.164 | 494.925           |                                   | passed   | passed   |
| resnet50 | offline      |     76.456 |       26.793 | -                 |                                   | passed   | passed   |
| resnet50 | singlestream |     76.456 |       16.303 | 61.337            |                                   | passed   | passed   |