To run this benchmark, first follow the setup steps in `closed/NVIDIA/README.md`. Then run the harness:

```
make run_harness RUN_ARGS="--benchmarks=rgat --scenarios=Offline --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=rgat --scenarios=Server --test_mode=PerformanceOnly"
```

For more details, please refer to `closed/NVIDIA/README.md`
