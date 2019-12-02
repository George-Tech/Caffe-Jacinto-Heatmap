# Caffe-Jacinto-Heatmap
Modify:
1. layer: heatmap_data_layer.cpp
2. new heatmapdataparam in caffe.proto
3. disable some if conditions for bndbox

Use:
put heatmap_data_layer.cpp in $caffe_root/src/caffe/layers/

put heatmap_data_layer.hpp in $caffe_root/include/caffe/layers/

replace caffe.proto in $caffe_root/src/caffe/proto/

replace bbox_util.cpp in $caffe_root/src/caffe/util/

Exp:
use transform_param | data_param the same way as SSD
heatmap_data_param:
label_map_file:""
heatmap_c: output channel
heatmap_h: output map height
heatmap_w: output map weight
heatmap_sigma: gaussin kernel R