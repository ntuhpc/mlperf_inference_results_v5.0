| Model    | Scenario     |   Accuracy |   Throughput | Latency (in ms)   | Power Efficiency (in samples/J)   | TEST01   | TEST04   |
|----------|--------------|------------|--------------|-------------------|-----------------------------------|----------|----------|
| resnet50 | multistream  |     76.456 |       15.466 | 517.256           |                                   | passed   | passed   |
| resnet50 | offline      |     76.456 |       23.162 | -                 |                                   | passed   | passed   |
| resnet50 | singlestream |     76.456 |       14.273 | 70.064            |                                   | passed   | passed   |