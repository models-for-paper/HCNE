import networkx as nx
import scipy.io
from collections import defaultdict
matlab_filename = ["./Amherst41.mat", "./Hamilton46.mat", "Georgetown15.mat"]

matlab_object = scipy.io.loadmat(matlab_filename[2])
scipy_sparse_graph = matlab_object["A"]
G = nx.from_scipy_sparse_matrix (scipy_sparse_graph)

attribute_dict = {
    # "id" : 0,
    "student_fac" : 0,
    "gender" : 1,
    "major_index" : 2, ##s
    "second_major" : 3,
    "dorm" : 4,
    "year" : 5,  ##f
    "high_school" : 6,
    }

def get_attribute_partition(matlab_object, attribute):
    attribute_rows = matlab_object["local_info"]
    # Indicies as defined in comment above                                      
    try:
        index = attribute_dict[attribute]
    except KeyError:
        raise KeyError("Given attribute " + attribute + " is not a valid choice.\nValid choices include\n" + str(attribute_dict.keys()))
    current_id = 0
    partition = defaultdict(set)
    for row in attribute_rows:
        if not(len(row) == 7):
            raise ValueError("Row " + str(current_id) + " has " + str(len(row)) + " rather than the expected 7 rows!")
        value = row[index]
        partition[value].add(current_id)
        current_id += 1
    return dict(partition)

for attribute in attribute_dict:
    partition = get_attribute_partition(matlab_object, attribute)
    for value in partition:
        for node in partition[value]:
            G.node[node][attribute] = value
years = set()
major = set()
sf = set()
gd = set()
sm = set()
dorm = set()
hs = set()
for node in G.node:
    sf.add(G.node[node]['student_fac'])
    gd.add(G.node[node]['gender'])
    years.add(G.node[node]['year'])
    major.add(G.node[node]['major_index'])
    sm.add(G.node[node]['second_major'])
    dorm.add(G.node[node]['dorm'])
    hs.add(G.node[node]['high_school'])

start_id = len(G.node)
years_nodes = [set() for i in range(len(years))]
year_node = []
i = 0
for year in years:
    for node in G.node:
        if G.node[node]['year'] == year:
            years_nodes[i].add(node)
    i = i+1

majors_nodes_t = [[set() for i in range(len(major))] for j in range(len(years))]

j = 0
for year in years_nodes:
    i = 0
    for m in major:
        for node in year:
            if G.node[node]['major_index'] == m:
                majors_nodes_t[j][i].add(node)
        i = i + 1
    
    j = j + 1

majors_nodes = [[] for j in range(len(years))]

i = 0
for mn in majors_nodes_t:
    for s in mn:
        if (len(s) != 0):
            majors_nodes[i].append(s)
    i = i + 1

majors_nodes_id = [[] for j in range(len(years))]
years_nodes_id = []
i = 0
for mn in majors_nodes:
    for s in mn:
        majors_nodes_id[i].append(start_id)
        start_id = start_id + 1
    i = i + 1

for ye in years_nodes:
    years_nodes_id.append(start_id)
    start_id = start_id + 1

root_id = start_id

G1 = nx.DiGraph()
for n in years_nodes_id:
    G1.add_edge(root_id, n)

i = 0
for y in years_nodes_id:
    for n in majors_nodes_id[i]:
        G1.add_edge(y, n)
    i = i + 1

i = 0
for mn in majors_nodes:
    j = 0
    for s in mn:
        for n in s:
            G1.add_edge(majors_nodes_id[i][j], n)
        j = j + 1
    i = i + 1

# write file
import pandas as pd
name = 'georgetown'


df = pd.DataFrame()
edges_xxx =  []
for e in G.edges:
    edges_xxx.append((e[0], e[1]))
df = df.append(edges_xxx)
df.to_csv(f"edges_{name}.txt", header=False, index=False, sep='\t')
print(len(G.node))
# add node num to file


df = pd.DataFrame()
tree2_xxx =  []
for e in G1.edges:
    tree2_xxx.append((e[0], e[1]))
df = df.append(tree2_xxx)
df.to_csv(f"tree2_{name}", header=False, index=False, sep='\t')
print(len(G1.node), ' ', len(G.node))
# add nodenum and leaf num


df = pd.DataFrame()
flag_xxx =  []
for n in G.node:
    flag_xxx.append((n, G.node[n]['student_fac']))
df = df.append(flag_xxx)
df.to_csv(f"flag_{name}.txt", header=False, index=False, sep='\t')
