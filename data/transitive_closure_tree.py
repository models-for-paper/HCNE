#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import re
# import regex as re
import pandas
from nltk.corpus import wordnet as wn
from tqdm import tqdm
try:
    wn.all_synsets
except LookupError as e:
    import nltk
    nltk.download('wordnet')

# make sure each edge is included only once
edges = set()

#    instance<---synset<---hyper...    -->(synset, hyper)
#    instance<---...hyper<-----h...       -->(instance, hyper)    (instance, h)
# 
nameset = dict()
'''
for synset in tqdm(wn.all_synsets(pos='n')):
    for hyper in synset.hypernyms():
        edges.add((synset.name(), hyper.name()))
'''
with open('./noun_filter.txt', 'r') as f:
    nn = eval(f.read())
for synset in tqdm(wn.all_synsets(pos='n')):
    f = False
    if synset.name() in nn:
        continue
    for hy in synset.closure(lambda s: s.hypernyms()):
        if (hy.name() in nn):
            f = True
    if f:
        continue
    for hyper in synset.hypernyms():
        edges.add((synset.name(), hyper.name()))


    # write the transitive closure of all hypernyms of a synset to file
    # for hyper in synset.closure(lambda s: s.hypernyms()):
    #     edges.add((synset.name(), hyper.name()))
        # 子类--父类
    # # also write transitive closure for all instances of a synset
    # for instance in synset.instance_hyponyms():
    #     for hyper in instance.closure(lambda s: s.instance_hypernyms()):
    #         edges.add((instance.name(), hyper.name()))
    #         for h in hyper.closure(lambda s: s.hypernyms()):
    #             edges.add((instance.name(), h.name()))

# （子，父）
nouns = pandas.DataFrame(list(edges), columns=['id1', 'id2'])
nouns['weight'] = 1

# Extract the set of nouns that have "mammal.n.01" as a hypernym
# 保留所有祖先位mamal的点的名字
mammal_set = set()
for synset in tqdm(wn.all_synsets(pos='n')):
    for hyper in synset.closure(lambda s: s.hypernyms()):
        if hyper.name() == 'mammal.n.01':
            mammal_set.add(synset.name())
            break
# mammal_set = set(nouns[nouns.id2 == 'mammal.n.01'].id1.unique())

mammal_set.add('mammal.n.01')

# Select relations that have a mammal as hypo and hypernym
# 保留所有mammal分支下的节点
mammals = nouns[nouns.id1.isin(mammal_set) & nouns.id2.isin(mammal_set)]

with open('mammals_filter.txt', 'r') as fin:
    filt = re.compile(f'({"|".join([l.strip() for l in fin.readlines()])})')
# print(filt)

filtered_mammals = mammals[~mammals.id1.str.cat(' ' + mammals.id2).str.match(filt)]
filtered_mammals = filtered_mammals[filtered_mammals['id1'] != 'pachyderm.n.01']
filtered_mammals = filtered_mammals[filtered_mammals['id2'] != 'pachyderm.n.01']
# nouns.to_csv('noun_closure.csv', index=False)
filtered_mammals.to_csv('mammal_closure.csv', index=False)

#进行节点归类
'''
OutEdgeDataView([('mammal.n.01', 'placental.n.01'), ('mammal.n.01', 'prototherian.n.01'), ('mammal.n.01', 'fossorial_mammal.n.01'), ('mammal.n.01', 'metatherian.n.01')])
'''
cls_name = {
    "placental.n.01" : 0,
    "prototherian.n.01" : 1,
    "fossorial_mammal.n.01" : 2,
    "metatherian.n.01" : 3
}
pre_names = set(filtered_mammals['id1']).union(filtered_mammals['id2'])
pre_names.remove('mammal.n.01')
pre_ids = []
pre_names = list(pre_names)
for pre in pre_names:
    sn = wn.synset(name=pre)
    find = False
    if (sn.name() in cls_name.keys()):
        pre_ids.append(cls_name[sn.name()])
        continue
    for hy in sn.closure(lambda s: s.hypernyms()):
        if hy.name() in cls_name.keys():
            pre_ids.append(cls_name[hy.name()])
            find = True
            break
    if not find:
        print(sn.name(), "Error")

mammals_cls = pandas.DataFrame({'id': pre_names, 'label': pre_ids})
mammals_cls.to_csv('mammal_label.csv', index=False)

'''
OutEdgeDataView([('entity.n.01', 'physical_entity.n.01'), ('entity.n.01', 'abstraction.n.06'), ('entity.n.01', 'thing.n.08')])
'''
cls_name = {
    "physical_entity.n.01" : 0,
    "abstraction.n.06" : 1,
    "thing.n.08" : 2
}
pre_names = set(nouns['id1']).union(nouns['id2'])
pre_names.remove('entity.n.01')
pre_ids = []
pre_names = list(pre_names)
for pre in pre_names:
    sn = wn.synset(name=pre)
    find = False
    if (sn.name() in cls_name.keys()):
        pre_ids.append(cls_name[sn.name()])
        continue
    for hy in sn.closure(lambda s: s.hypernyms()):
        if hy.name() in cls_name.keys():
            pre_ids.append(cls_name[hy.name()])
            find = True
            break
    if not find:
        print(sn.name(), "Error")

noun_cls = pandas.DataFrame({'id': pre_names, 'label': pre_ids})
# noun_cls.to_csv('noun_label.csv', index=False)
