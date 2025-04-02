import sys
import os
from importlib import import_module
import pkgutil
# Add the parent directory of your module to sys.path
base_path = "/mnt/data/rwang/tmp/closed/NVIDIA"
sys.path.append(base_path)


base_path = "/mnt/data/rwang/tmp/closed/NVIDIA/code"
sys.path.append(base_path)


print(os.path.isdir("/mnt/data/rwang/tmp/closed/NVIDIA/code"))
print(os.path.isdir("/mnt/data/rwang/tmp/closed/NVIDIA/code/harness"))
print(os.path.isfile("/mnt/data/rwang/tmp/closed/NVIDIA/code/harness/__init__.py"))

print("Looking for 'code' in sys.path:", sys.path)
print("Is 'code' a directory?", os.path.isdir("/mnt/data/rwang/tmp/closed/NVIDIA/code"))


import code
# Dynamically load the module
module_name = "code"
module = import_module(module_name)
print([name for _, name, _ in pkgutil.iter_modules(['code'])])
print(code.__file__)

print(f"Successfully imported {module_name}")



# Now try importing with the modified module name
module_name = "llama2-70b.tensorrt.dataset"  # Use underscores instead of hyphens
dataset_cls = getattr(import_module(module_name), "LlamaDataset")

print(f"Successfully imported {module_name}")

