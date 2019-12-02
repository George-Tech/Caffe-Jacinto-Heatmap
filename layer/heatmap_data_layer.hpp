#ifndef CAFFE_HEATMAP_DATA_LAYER_HPP_
#define CAFFE_HEATMAP_DATA_LAYER_HPP_

#include <string>
#include <map>
#include <vector>
#include <atomic>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
class HeatmapDataLayer : public DataLayer<Ftype, Btype> {
 public:
  HeatmapDataLayer(const LayerParameter& param, size_t solver_rank);
  void DataLayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) override;

  const char* type() const override {
    return "HeatmapData";
  }
  int ExactNumBottomBlobs() const override {
    return 0;
  }
  int MinTopBlobs() const override {
    return 1;
  }

 protected:
  void load_batch(Batch* batch, int thread_id, size_t queue_id) override;
  void start_reading() override {
    areader_->start_reading();
  }

  std::shared_ptr<DataReader<AnnotatedDatum>> sample_areader_, areader_;
  bool has_anno_type_;
  AnnotatedDatum_AnnotationType anno_type_;
  vector<BatchSampler> batch_samplers_;
  string label_map_file_;
  
  int datamap_c_;
  int datamap_h_;
  int datamap_w_;
  
  int heatmap_c_;
  int heatmap_h_;
  int heatmap_w_;
  int visualise_;
  float sigma_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
