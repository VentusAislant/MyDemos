### Install

1. Create Virtual Environment:
```shell
conda create -n aislant_demos python=3.10 -y
conda activate aislant_demos

```

2. Install packages
```shell
# for all(general)
pip install -e .

# for llm demos
pip install -e .[llm_demos]

# for gradio demos
pip install -e .[gradio_demos]

# for vision demos
pip install -e .[vision_demos]

```