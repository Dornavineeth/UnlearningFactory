<div align="center">

# Reproduce Results

</div>

The scripts below execute standard baseline unlearning experiments on the TOFU and MUSE datasets, evaluated using their corresponding benchmarks. 
```bash
bash scripts/tofu_unlearn.sh
bash scripts/muse_unlearn.sh
```

## Results



For all the experiments below, we used the following setup

| **Category**            | **Details** |
|-------------------------|------------|
| **Hardware**           | 2 × L40s GPUs (48GB each) |
| **Distributed Computing** | [DeepSpeed ZeRO Stage 3 (Accelerate)](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed) |
| **Hyperparameters**    | Learning Rate (lr) = 1e-5 <br> α = 1, γ = 1, β = 0.1 (where applicable) <br> Number of Epochs = 10 <br> Optimizer: [paged_adamw_32bit](https://huggingface.co/docs/bitsandbytes/main/en/reference/optim/adamw#bitsandbytes.optim.PagedAdamW) |

__NOTE__: Results may vary even with the same effective hyperparameters when trained on a single GPU. **Please use these numbers only for reproducibility purposes**. Some methods, such as SimNPO, can be significantly improved with careful tuning.

### TOFU  unlearning on `Llama-2-7b-hf-chat`


<style>
  table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 20px;
    overflow-x: auto;
    white-space: nowrap;
  }
  th, td {
    border: 1px solid #000;
    padding: 4px;
    word-wrap: break-word;
    word-break: break-all;
    text-align: center;
  }
  th {
    text-align: center;
  }
  col.argument {
    width: 30%;
  }
  col.description {
    width: 70%;
  }
</style>



<div style="overflow-x: auto; max-width: 100%;">
<table class="dataframe">
  <thead>
    <tr>
      <th>Method</th>
      <th colspan="3" halign="left">forget01</th>
      <th colspan="3" halign="left">forget05</th>
      <th colspan="3" halign="left">forget10</th>
    </tr>
    <tr>
      <th></th>
      <th>forget_quality</th>
      <th>model_utility</th>
      <th>forget_truth_ratio</th>
      <th>forget_quality</th>
      <th>model_utility</th>
      <th>forget_truth_ratio</th>
      <th>forget_quality</th>
      <th>model_utility</th>
      <th>forget_truth_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Finetuned</th>
      <td>1.27e-03</td>
      <td>6.28e-01</td>
      <td>5.31e-01</td>
      <td>1.33e-13</td>
      <td>6.28e-01</td>
      <td>5.12e-01</td>
      <td>4.35e-25</td>
      <td>6.28e-01</td>
      <td>5.19e-01</td>
    </tr>
    <tr>
      <th>Retain</th>
      <td>0.00e+00</td>
      <td>6.26e-01</td>
      <td>6.77e-01</td>
      <td>0.00e+00</td>
      <td>6.27e-01</td>
      <td>6.70e-01</td>
      <td>0.00e+00</td>
      <td>6.13e-01</td>
      <td>6.81e-01</td>
    </tr>
    <tr>
      <td colspan="20"> </td>
    </tr>
    <tr>
      <th>GradAscent</th>
      <td>1.88e-04</td>
      <td>5.47e-01</td>
      <td>3.63e-01</td>
      <td>1.94e-119</td>
      <td>0.00e+00</td>
      <td>8.82e-96</td>
      <td>1.06e-239</td>
      <td>0.00e+00</td>
      <td>2.21e-32</td>
    </tr>
    <tr>
      <th>GradDiff</th>
      <td>3.02e-03</td>
      <td>5.73e-01</td>
      <td>4.09e-01</td>
      <td>1.94e-119</td>
      <td>5.56e-01</td>
      <td>4.14e-95</td>
      <td>1.80e-229</td>
      <td>5.81e-01</td>
      <td>1.46e-07</td>
    </tr>
    <tr>
      <th>IdkDPO</th>
      <td>9.71e-02</td>
      <td>5.63e-01</td>
      <td>6.68e-01</td>
      <td>4.02e-06</td>
      <td>3.65e-02</td>
      <td>6.69e-01</td>
      <td>5.42e-13</td>
      <td>4.13e-02</td>
      <td>6.44e-01</td>
    </tr>
    <tr>
      <th>NPO</th>
      <td>4.05e-01</td>
      <td>5.83e-01</td>
      <td>6.54e-01</td>
      <td>8.78e-02</td>
      <td>5.32e-01</td>
      <td>7.11e-01</td>
      <td>4.16e-01</td>
      <td>5.37e-01</td>
      <td>7.26e-01</td>
    </tr>
    <tr>
      <th>SimNPO</th>
      <td>1.27e-03</td>
      <td>5.78e-01</td>
      <td>4.14e-01</td>
      <td>1.06e-106</td>
      <td>5.98e-01</td>
      <td>3.94e-05</td>
      <td>1.47e-198</td>
      <td>5.96e-01</td>
      <td>3.17e-04</td>
    </tr>
  </tbody>
</table>
</div>


### TOFU  unlearning on `Llama-3.2-1B-Instruct`

<div style="overflow-x: auto; max-width: 100%;">
<table class="dataframe">
  <thead>
    <tr>
      <th>Method</th>
      <th colspan="3" halign="left">forget01</th>
      <th colspan="3" halign="left">forget05</th>
      <th colspan="3" halign="left">forget10</th>
    </tr>
    <tr>
      <th></th>
      <th>forget_quality</th>
      <th>model_utility</th>
      <th>forget_truth_ratio</th>
      <th>forget_quality</th>
      <th>model_utility</th>
      <th>forget_truth_ratio</th>
      <th>forget_quality</th>
      <th>model_utility</th>
      <th>forget_truth_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Finetuned</th>
      <td>1.43e-02</td>
      <td>5.98e-01</td>
      <td>4.75e-01</td>
      <td>2.96e-13</td>
      <td>5.98e-01</td>
      <td>4.73e-01</td>
      <td>8.08e-22</td>
      <td>5.98e-01</td>
      <td>4.76e-01</td>
    </tr>
    <tr>
      <th>Retain</th>
      <td>0.00e+00</td>
      <td>5.96e-01</td>
      <td>6.48e-01</td>
      <td>0.00e+00</td>
      <td>5.98e-01</td>
      <td>6.34e-01</td>
      <td>0.00e+00</td>
      <td>5.93e-01</td>
      <td>6.28e-01</td>
    </tr>
    <tr>
      <td colspan="20"> </td>
    </tr>
    <tr>
      <th>GradAscent</th>
      <td>2.66e-01</td>
      <td>3.25e-01</td>
      <td>5.91e-01</td>
      <td>1.94e-119</td>
      <td>0.00e+00</td>
      <td>2.52e-23</td>
      <td>1.06e-239</td>
      <td>0.00e+00</td>
      <td>2.25e-18</td>
    </tr>
    <tr>
      <th>GradDiff</th>
      <td>7.66e-01</td>
      <td>4.28e-01</td>
      <td>5.74e-01</td>
      <td>1.94e-119</td>
      <td>5.35e-01</td>
      <td>3.87e-34</td>
      <td>1.06e-239</td>
      <td>4.91e-01</td>
      <td>3.53e-27</td>
    </tr>
    <tr>
      <th>IdkDPO</th>
      <td>1.43e-02</td>
      <td>5.06e-01</td>
      <td>5.96e-01</td>
      <td>1.12e-05</td>
      <td>6.82e-02</td>
      <td>6.22e-01</td>
      <td>4.64e-12</td>
      <td>2.35e-01</td>
      <td>5.99e-01</td>
    </tr>
    <tr>
      <th>NPO</th>
      <td>9.19e-01</td>
      <td>5.61e-01</td>
      <td>6.60e-01</td>
      <td>1.42e-01</td>
      <td>4.53e-01</td>
      <td>7.03e-01</td>
      <td>1.58e-02</td>
      <td>4.64e-01</td>
      <td>6.98e-01</td>
    </tr>
    <tr>
      <th>SimNPO</th>
      <td>5.79e-01</td>
      <td>4.59e-01</td>
      <td>5.45e-01</td>
      <td>5.01e-100</td>
      <td>5.80e-01</td>
      <td>4.19e-03</td>
      <td>2.47e-203</td>
      <td>5.43e-01</td>
      <td>1.07e-05</td>
    </tr>
  </tbody>
</table>
</div>


### MUSE  unlearning on `Llama-2-7b-hf`

<div style="overflow-x: auto; max-width: 100%;">
<table class="dataframe">
  <thead>
    <tr>
      <th>Method</th>
      <th colspan="4" halign="left">News</th>
      <th colspan="4" halign="left">Books</th>
    </tr>
    <tr>
      <th></th>
      <th>forget_knowmem_ROUGE</th>
      <th>forget_verbmem_ROUGE</th>
      <th>privleak</th>
      <th>retain_knowmem_ROUGE</th>
      <th>forget_knowmem_ROUGE</th>
      <th>forget_verbmem_ROUGE</th>
      <th>privleak</th>
      <th>retain_knowmem_ROUGE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Finetuned</th>
      <td>6.44e-01</td>
      <td>5.76e-01</td>
      <td>-9.98e+01</td>
      <td>5.55e-01</td>
      <td>4.71e-01</td>
      <td>9.97e-01</td>
      <td>-5.73e+01</td>
      <td>6.91e-01</td>
    </tr>
    <tr>
      <th>Retain</th>
      <td>3.34e-01</td>
      <td>2.06e-01</td>
      <td>-4.54e+00</td>
      <td>5.59e-01</td>
      <td>3.04e-01</td>
      <td>1.41e-01</td>
      <td>7.96e+00</td>
      <td>6.86e-01</td>
    </tr>
    <tr>
      <td colspan="20"> </td>
    </tr>
    <tr>
      <th>GradAscent</th>
      <td>0.00e+00</td>
      <td>0.00e+00</td>
      <td>5.21e+01</td>
      <td>0.00e+00</td>
      <td>0.00e+00</td>
      <td>0.00e+00</td>
      <td>-6.67e-01</td>
      <td>0.00e+00</td>
    </tr>
    <tr>
      <th>GradDiff</th>
      <td>4.10e-01</td>
      <td>8.92e-03</td>
      <td>9.32e+01</td>
      <td>3.72e-01</td>
      <td>1.76e-01</td>
      <td>1.64e-01</td>
      <td>-3.78e+01</td>
      <td>3.00e-01</td>
    </tr>
    <tr>
      <th>NPO</th>
      <td>5.57e-01</td>
      <td>3.49e-01</td>
      <td>-8.60e+01</td>
      <td>5.13e-01</td>
      <td>3.23e-01</td>
      <td>8.42e-01</td>
      <td>-5.42e+01</td>
      <td>5.52e-01</td>
    </tr>
    <tr>
      <th>SimNPO</th>
      <td>5.35e-01</td>
      <td>3.60e-01</td>
      <td>-8.61e+01</td>
      <td>5.09e-01</td>
      <td>3.24e-01</td>
      <td>8.42e-01</td>
      <td>-5.43e+01</td>
      <td>5.42e-01</td>
    </tr>
  </tbody>
</table>
</div>