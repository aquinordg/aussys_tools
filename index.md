# Package rdga_4k

Python package [rdga_4k](https://github.com/aquinordg/rdga_4k): Random Data Generator Algorithm for Clustering.

The package generates synthetic data for applications in clustering algorithms.

## Functions

### Categorical Binary Random Data
```markdown

catbird(m, n, k, lmbd=0.5, eps=0.5, random_state=None)

```
Generates random categorical binary data with _m_ examples (rows), _n_ attributes (columns), for _k_ clusters. The algorithm divides the number of examples into equal amounts within each of the clusters.

Given s = n//2 + 1, the construction of the databases, the process is divided into three phases:

- We filled in _s_ features, obtained from the multiplication of a single probability matrix, for each cluster, and a array, for each example. Both the matrix and the array values are obtained from gaussian distributions.

- The remaining _n - s_ features are filled using the _eps_ interference value. Examples within each cluster receive interference in the same positions, uniquely.

- In this way, we apply a sigmoid function to the database and convert it to the binary base, given the probability _lmbd_. Where we associate 1 to values smaller than the parameter.


#### Parameters

**m**: _int_<br/>
Number of examples or rows.

**n**: _int_<br/>
Number of features or columns.

**k**: _int_<br/>
Number of clusters.

**lmbd**: _float in range [0.0, 1.0]_<br/>
Reference probability used in the transformation to the binary base of the database, after applying the sigmoid function.

**eps**: _float in range [0.0, 1.0]_<br/>
Interference used in the particularization of clusters, before the application of the sigmoid function and the transformation to the binary base.

**random_state**: _int or None_<br/>
Random generator seed, useful for creating reproducible databases.

#### _Returns_

**data**: _array-like of shape (m, n)_<br/>
Output database.

**labels**: _array-like of shape (1, m)_<br/>
Output database labels.

####  Example

```markdown

catbird(m=5, n=3, k=2)

```
_**Cluster 0**_

_Cluster matrix (W0):_ [[2.46 1.38], [0.34 1.02]]<br/>
_Example array * cluster matrix (A0 * W0):_ [0.16 1.65] * [[2.46 1.38], [0.34 1.02]] = [0.98, 1.92]<br/>
_A0 * W0 with eps after sigmoid function:_ [0.50, 0.72, 0.87]<br/>
_A0 * W0 after binarization:_ [0, 0, 0]<br/>

_Cluster matrix (W0):_ [[2.46 1.38], [0.34 1.02]]<br/>
_Example array * cluster matrix (A1 * W0):_ [ 0.66 -0.22] * [[2.46 1.38], [0.34 1.02]] = [1.56, 0.68]<br/>
_A1 * W0 with eps after sigmoid function:_ [0.5, 0.82, 0.66]<br/>
_A1 * W0 after binarization:_ [0, 0, 0]<br/>

_Cluster matrix (W0):_ [[2.46 1.38], [0.34 1.02]]<br/>
_Example array * cluster matrix (A2 * W0):_ [-1.12 -0.63] * [[2.46 1.38], [0.34 1.02]] = [-3.00, -2.21]<br/>
_A2 * W0 with eps after sigmoid function:_ [0.5, 0.04, 0.09]<br/>
_A2 * W0 after binarization:_ [0, 1, 1]<br/>

_**Cluster 1**_

_Cluster matrix (W1):_ [[0.31 -1.22], [-0.22  1.33]]<br/>
_Example array * cluster matrix (A3 * W1):_ [0.02 1.98] * [[0.31 -1.22], [-0.22  1.33]] = [-0.43, 2.62]<br/>
_A3 * W1 with eps after sigmoid function:_ [0.39, 0.93, 0.5]<br/>
_A3 * W1 after binarization:_ [1, 0, 0]  

_Cluster matrix (W1):_ [[2.46 1.38], [0.34 1.02]]<br/>
_Example array * cluster matrix (A4 * W1):_ [ 1.44  -0.28] * [[0.31 -1.22], [-0.22  1.33]] = [0.51, -2.15]<br/>
_A4 * W1 with eps after sigmoid function:_ [0.62, 0.10, 0.5]<br/>
_A4 * W1 after binarization:_ [0, 1, 0]<br/>

**data:** [[0, 0, 0], [0, 0, 0], [0, 1, 1], [1, 0, 0], [0, 1, 0]]

**labels:** [0 0 0 1 1]

#### Install

```markdown
pip install git+https://github.com/aquinordg/rdga_4k.git
```

#### Contact us

E-mail: aquinordga@gmail.com
