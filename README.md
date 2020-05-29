# Code for HCNE

## Demo

```shell
sh run.sh
```

## Data conversion script
* To GCN models
insert into the source code
```python
from convert_to_gcn import load_tree_sr, mask_data, accuracy_score
name='wordnet'
target, A, num_nodes = load_tree_sr(name)
test_rate = 0.1
train_data, test_data = mask_data(test_rate, num_nodes, target)
train_data = train_data.long()
test_data = test_data.long()
```

* To GrapgSAGE
insert into the source code
```python
from convert_to_graphsage import load_tree_sr
np.random.seed(22)
random.seed(22)
# num_nodes = 2708
feat_data, labels, adj_lists, num_nodes = load_tree_sr('wordnet')
features = nn.Embedding(num_nodes, 1)
features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
```

* To Poincare Embeddings
insert into the source code
```shell
python convert_to_pe.py
```

## Raw data parser
* For Facebook-100 dataset
run in shell
```shell
python data/parse_facebook100.py
```

*For wordnet dataset
run in shell
```shell
python transitive_closure_tree.py
```
## Dataset link

<a href='https://raw.githubusercontent.com/ab2525/ia-archiveteam/c2b56dd7f2c50899df74b02830019badb9a2a445/oxford-2005-facebook-matrix/facebook100.zip'>Facebook 100 </a>

<a href=https://ia800504.us.archive.org/1/items/oxford-2005-facebook-matrix/facebook100.zip>Facebook 100 (1) </a>

<a href='https://github.com/wordnet/wordnet'> WordNet </a>












