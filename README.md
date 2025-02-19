### Install

- 1. Create Virtual Environment:
```shell
conda create -n aislant python=3.10 -y
conda activate aislant

```

- 2. Install packages
```shell
# for all(general)
pip install -e .

# for ai(only for this module)
pip install -e .[ai]

# for various_tools(only for this module)
pip install -e .[visual_tools]
```