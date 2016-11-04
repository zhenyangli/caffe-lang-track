#ifndef CAFFE_REVERSE_LAYER_HPP_
#define CAFFE_REVERSE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Reverse the input Blob.
 *
 * This layer reverses the input Blob with or without the second input Blob.
 * Without the second input Blob, it reverses the entire input Blob.
 * @f$ [x_1^1, ..., x_1^N] @f$ to @f$ [x_1^N, ..., x_1^1] @f$
 * With the second input Blob, it reverses examples in between the reverse
 * segments (pairs of start index and length) specified in the second Blob.
 * The remaining elements in the second Blob can be padded with negative values
 * to indicate no more reverse segment exists.
 * For example, if the second Blob is @f$ [0,3,4,3,-1,-1] @f$ then
 * @f$ [x_1^1,x_1^2,x_1^3,x_1^4,x_1^5,x_1^6,x_1^7,x_1^8,x_1^9] @f$ will be
 * @f$ [x_1^3,x_1^2,x_1^1,x_1^4,x_1^7,x_1^6,x_1^5,x_1^8,x_1^9] @f$
 */
template <typename Dtype>
class ReverseLayer : public Layer<Dtype> {
 public:
  explicit ReverseLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Reverse"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    // Cannot propagate to reverse segment Blob.
    return bottom_index !=1;
  }

 protected:
  /**
   * @param bottom input Blob vector (length 1-2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x_1 @f$
   *   -# @f$ (M \times 1 \times 1 \times 1) @f$
   *      the inputs @f$ x_2 @f$ -- Reverse segment pairs (start, length)
   *      indicating where to reverse in @f$ x_1 @f$.
   *      Examples in between x_2[i*2], x_2[i*2]+x_2[i*2+1]-1 will be reversed.
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the outputs -- It should be
   *      -(w/o @f$ x_2 @f$) Blob of reversed examples of the entire input
   *      -(w/ @f$ x_2 @f$) Blob of reversed examples in the input segments
   *      specified by @f$ x_2 @f$ and identical examples in the remaining
   *      input segments
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /**
   * @brief Computes the error gradient w.r.t. the input.
   *
   * @param top output Blob vector (length 1), providing the error gradient
   *        with respect to the outputs
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 1-2), into which the top error 
   *        gradient is reversed.
   *   - This layer cannot backprop to @f$ x_2 @f$.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // compute and store offsets of reverse pairs
  void reverse_offsets(const vector<pair<int, int> >& reverse_segment);

  void reverse(const Dtype* const src, Dtype* const dst);

  Blob<Dtype> temp_;
  Blob<int> reverse_offset_;
  int reverse_unit_size_;
  int num_reverse_pairs_;
};

}  // namespace caffe

#endif  // CAFFE_REVERSE_LAYER_HPP_
