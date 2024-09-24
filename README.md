## Setup LLamaFactory

```
conda create --name llama python=3.11
cd LLamaFactory
conda activate llama
pip install -e ".[torch,metrics]"
pip install bitsandbytes
```