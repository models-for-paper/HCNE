data_dir='./data/'
# hidden_dim=32
hidden_dim=16
# radius_scale_factor=0.9
radius_scale_factor=0.9
class_num=6
coor_dim=2

batch_num=4096
epoch_num=500
learning_rate=1e-3

radius_0=10

graph_file='edges_amherst.txt'
tree_file='tree2_amherst'

python train.py --data_dir $data_dir --hidden_dim $hidden_dim --radius_scale_factor $radius_scale_factor --class_num $class_num --coor_dim $coor_dim --batch_num $batch_num --epoch_num $epoch_num --learning_rate $learning_rate --radius_0 $radius_0 --graph_file $graph_file --tree_file $tree_file
