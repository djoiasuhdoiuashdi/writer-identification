# writer-identification

## Steps

### SIFT generation

2 Different unsupervised SIFT descriptor approaches -> generate SIFT descriptors

### SIFT transformation
Using Vectors of Locally Aggregated Descriptors (VLAD)
local descriptors -> global descriptors (100 k-means clustering)

Generalized Max Pooling (GMP) iwth regularization parameter gamma = 1000
power-normalized with power of 0.5 and l²-normalized
This process is repeated 5 times 

These 5 are jointly PCA-whitened and again l²-normalized.

This final representation is used with similarity comparison using cosine distance.
