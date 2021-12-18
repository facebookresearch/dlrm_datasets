# Embedding lookup synthetic dataset
## Description
A synthetic dataset comprised of input data for embedding lookup layers arising
in recommendation models, such as [DLRM](https://github.com/facebookresearch/dlrm),
that shares memory access reuse patterns similar to those arising in Meta
production recommendation workloads.

```
 Emb_out(F)                 Emb_in(E)    Indices(I)  Lengths(L)  Weights(W) [optional]
------------         /--  ------------       -          -            -    --\
|          |         |    |          |      | |        | |          | |     |
|          |  =  Op <     |          |  ,   | |   ,    | |   ,      | |      >
|          |         |    |          |      | |        | |          | |     |
------------         |    |          |      | |         -           | |     |
                     |    |          |      | |                     | |     |
                     |    ____________      | |                     | |     |
                     |                      | |                     | |     |
                     |                      | |                     | |     |
                     \--                     _                       _    --/
```

Each output embedding is the result of combining input embedding at locations
specified by values in the Indices vector and aggregated according to the values
in the Lengths vector (optionally weighted by values in the Weights vector).

Mathematically, each output embedding vector is computed as
```
F_i = \sum_{j \in range(L_i) + P_{i}} E_{I_j} * W_{I_j}
```
where `P_i` denotes the prefix sum of `L` up to index `i` (or "Offsets"): `P_i = \sum_{j \in range(i-1)} L_j`.
In practice, the Lengths vector is stored in the form of Offsets to reduce complexity of the operator.

## Usage
The synthetic dataset provided in this project serves as sample inputs for the Indices and Offsets vectors;
the corresponding Lengths vector is provided for correctness validation as well. Each `pt` file contains an
independently generated synthetic dataset with batch size and the number of tables specified in the filename.
For example, the dataset `fbgemm_t856_bs65536.pt` represents a single batch of 65536 samples for 856 tables.
To load the synthetic dataset,

```python
import torch

indices, offsets, lengths = torch.load("../dlrm_datasets/embedding_bag/fbgemm_t856_bs65536.pt")
```

The intent of this data is to support researchers and system designers with data representative of the memory access patterns observed during training of Meta's production ads models in order to offer guidance for their work in improving software computing solutions and hardware design.

### FBGemm
These datasets serve as input to the [`split_table_batched_embeddings`](https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/bench/split_table_batched_embeddings_benchmark.py) 
benchmark, a part of the [FBGemm project](https://github.com/pytorch/FBGEMM).  For those interested
in benchmarking subsets of the tables provided in the dataset, a 
[batch execution script](https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu/bench/scripts)  has 
been added to the project as well.

## Reuse pattern
Datasets in this project are accompanied by the observed reuse factor of unique
indices found in the dataset, represented as a histogram. Consider the following
histogram of reuse factors:

```
Reuse factor:   Proportion of data in bin
(0, 1]:         0.069
(1, 2]:         0.044
(2, 4]:         0.068
(4, 8]:         0.101
(8, 16]:        0.121
(16, 32]:       0.104
(32, 64]:       0.073
(64, 128]:      0.058
(128, 256]:     0.052
(256, 512]:     0.050
(512, 1024]:    0.049
(1024, 2048]:   0.048
(2048, 4096]:   0.048
(4096, 8192]:   0.043
(8192, 16384]:  0.031
(16384, 32768]: 0.023
(32768+:        0.019
```

Each bin reflects a range of reuse factors and the value corresponds to the
proportion of data found to have the given reuse factor. Datasets in this project
have been chosen as their reuse factor distributions reflect those found in production
workloads.

Lengths vector values are not held to as high of a standard; however, synthetic data for this vector
is also designed to align closely with values arising in production.

## Source data
The production data used as source for this synthetic data is post-hashed data and has been de-identified to
further preserve its integrity.

License
-------
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
