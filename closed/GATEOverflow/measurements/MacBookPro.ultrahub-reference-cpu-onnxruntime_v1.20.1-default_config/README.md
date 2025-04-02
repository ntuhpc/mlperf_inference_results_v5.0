| Model    | Scenario     |   Accuracy |   Throughput | Latency (in ms)   | Power Efficiency (in samples/J)   | TEST01   | TEST04   |
|----------|--------------|------------|--------------|-------------------|-----------------------------------|----------|----------|
| resnet50 | multistream  |     76.456 |       35.066 | 228.14            |                                   | passed   | passed   |
| resnet50 | singlestream |     76.456 |       28.515 | 35.069            |                                   | passed   | passed   |
| resnet50 | offline      |     76.456 |       66.339 | -                 |                                   | passed   | passed   |