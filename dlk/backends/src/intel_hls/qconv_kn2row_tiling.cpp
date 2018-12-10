/* Copyright 2018 The Blueoil Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdio>
#include "intel_hls/config.h"

namespace p = conv_kn2row_params;

T_out_hls PE_kn2row_tiling(uint32 k_buf, uint32 in_buf[2])
{
  T_out_hls out = 0;
  uint6 nk_buf_ppcounut = __builtin_popcount(~k_buf);

#pragma unroll
  for (unsigned ib = 0; ib < p::max_in_b; ib++) {
    // TODO: we can save resources by precomputing the sum of ~k_buf.
    uint6 xnor_popcount = __builtin_popcount(~(in_buf[ib] ^ k_buf));
    out += ((xnor_popcount - nk_buf_ppcounut) << ib);
  }
  return out;
}

hls_avalon_slave_component void intel_hls_qconv_kn2row_tiling_impl(
  hls_avalon_slave_register_argument
    ihc::mm_master<T_in_hls, ihc::aspace<1>, ihc::awidth<32>, ihc::dwidth<BW_>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &in_data,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_out_hls, ihc::aspace<2>, ihc::awidth<32>, ihc::dwidth<BW_>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &out_data,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_k_hls, ihc::aspace<3>, ihc::awidth<32>, ihc::dwidth<BW_>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &k_data,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_out_hls, ihc::aspace<4>, ihc::awidth<32>, ihc::dwidth<BW_>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &threshold_data,
  hls_avalon_slave_register_argument int32 in_w, hls_avalon_slave_register_argument int32 in_h,
  hls_avalon_slave_register_argument int32 in_c_by_word, hls_avalon_slave_register_argument int32 out_w,
  hls_avalon_slave_register_argument int32 out_h, hls_avalon_slave_register_argument int32 out_c,
  hls_avalon_slave_register_argument int32 k_w, hls_avalon_slave_register_argument int32 k_h,
  hls_avalon_slave_register_argument int32 pad, hls_avalon_slave_register_argument int32 use_threshold)
{
  /// just alias for better understanding
  static const unsigned out_c_low = p::num_pe;

OUTSIDE_TILE_LOOP:
#pragma unroll 1
#pragma max_concurrency 1
#pragma loop_coalesce 2
  for (int ih_high = 0; ih_high < in_h + 2 * pad; ih_high += p::tile_h) {
    for (int iw_high = 0; iw_high < in_w + 2 * pad; iw_high += p::tile_w) {
      // in_buf shoule be banked by 8 elems, because this has 2 bits per an element, and
      // 4 inputs are computed along with input channel dimension at a cycle.
      // This also should be doublepump because this loads next data from bus while computing the others.
      hls_memory hls_singlepump hls_bankbits(0, 1)
        T_in_hls in_buf[p::in_tile_h][p::in_tile_w][p::max_in_c_by_word][p::max_in_b];

    INPUT_LOAD_MODULE:
#pragma unroll 1
#pragma max_concurrency 1
#pragma loop_coalesce 2
      for (int ih_low = 0; ih_low < p::in_tile_h; ++ih_low) {
        for (int iw_low = 0; iw_low < p::in_tile_w; ++iw_low) {
          /// index must care the padding, so we skip the padding part that
          /// doesn't exist in actual memory.
          const int ih = (ih_low + ih_high - pad);
          const int iw = (iw_low + iw_high - pad);
          const bool input_on = (ih >= 0) && (iw >= 0) && (ih < in_h) && (iw < in_w);

#pragma unroll 2
#pragma max_concurrency 2
          for (int ic = 0; ic < in_c_by_word; ic++) {
#pragma unroll
#pragma max_concurrency 2
            for (int ib = 0; ib < p::max_in_b; ib++) {
              const int _in_w = int(in_w);
              const int _in_c = int(in_c_by_word);
              // loading inputs from bus.
              // if the coordinates on the padding, this stores 0 instead of loading the data.
              in_buf[ih_low][iw_low][ic][ib] =
                (input_on)
                  ? in_data[ih * _in_w * _in_c * p::max_in_b + iw * _in_c * p::max_in_b + ic * p::max_in_b + ib]
                  : T_in_hls(0);
            }
          }
        }
      }

#pragma unroll 1
#pragma max_concurrency 1
      for (int oc_high = 0; oc_high < out_c; oc_high += out_c_low) {
        // out_buf shoule be banked by 16 elems, because out_c_low is 16, which log2 is 4.
        // This also should should be doublepump, because accumulation happens at every cycle,
        // requiring reading a data and computing it, then rewriting it to the same address.
        hls_memory hls_doublepump hls_bankbits(0, 1, 2) T_out_hls out_buf[p::tile_w][p::tile_w][out_c_low];

        // k_buf shoule be banked by 64 elems, because
        // 16 kernels are needed to produce 16 outputs which is fully banked by 16 on out_c_low
        // Also per 1 output, 4 kernels are additionally needed to product with the 8 inputs coming from bus.
        // Only singlepump is OK for kernel.
        hls_memory hls_singlepump hls_bankbits(0, 1, 2, 3)
          hls_memory T_k_hls k_buf[p::max_k_h][p::max_k_w][p::max_in_c_by_word][out_c_low];

        // threshold loading module.
        // just relay on automatic unroll.
        hls_memory hls_singlepump hls_bankbits(0, 1, 2, 3, 4) T_out_hls threshold_buf[out_c_low][p::num_thresholds];

      KERNEL_LOAD_MODULE:
#pragma unroll 1
#pragma max_concurrency 1
#pragma loop_coalesce 2
        for (int kh = 0; kh < k_h; ++kh) {
          for (int kw = 0; kw < k_w; ++kw) {
#pragma unroll 2
#pragma max_concurrency 2
            for (int ic = 0; ic < in_c_by_word; ic++) {
#pragma unroll
#pragma max_concurrency 8
              for (int oc = 0; oc < out_c_low; oc++) {
                /// currently kernel order is NoHWCNi, which means the
                /// outermost dimension "N" is split into 2 high and low
                /// parts. we should be carefull when compute the index.
                const int _in_c = int(in_c_by_word);
                const int _out_c = int(out_c);
                const int _k_w = int(k_w);
                const int _k_h = int(k_h);
                const int idx_k = (kh * _k_w * _in_c * out_c_low) + (kw * _in_c * out_c_low) + (ic * out_c_low) + oc +
                                  (oc_high * _k_h * _k_w * _in_c);
                k_buf[kh][kw][ic][oc] = k_data[idx_k];
              }
            }
          }
        }

      THRESHOLD_LOAD_UNIT:
        if (use_threshold > 0) {
#pragma unroll
#pragma max_concurrency 8
          for (unsigned oc = 0; oc < out_c_low; oc++) {
#pragma unroll
#pragma max_concurrency 4
            for (unsigned i = 0; i < p::num_thresholds; i++) {
              unsigned idx_th = (oc_high + oc) * p::num_thresholds + i;
              threshold_buf[oc][i] = threshold_data[idx_th];
            }
          }
        }

      KN2ROW_KERNEL_LOOP:
#pragma unroll 1
#pragma max_concurrency 1
#pragma loop_coalesce 2
        for (int kh = 0; kh < k_h; ++kh) {
          for (int kw = 0; kw < k_w; ++kw) {

            // computation module over the tile
          MAC_COMPUTATION_MODULE:
#pragma loop_coalesce 2
#pragma unroll 1
#pragma max_concurrency 1
            for (int ih = 0; ih < p::in_tile_h; ++ih) {
              for (int iw = 0; iw < p::in_tile_w; ++iw) {
                const int oh = ih - kh;
                const int ow = iw - kw;
                const bool output_on = (oh >= 0) && (ow >= 0) && (oh < p::tile_h) && (ow < p::tile_w);
                // now, num_pe == 8
                hls_register T_out_hls out_regs[out_c_low] = {0, 0, 0, 0, 0, 0, 0, 0};

                // MAC compute module.
#pragma unroll 2
#pragma max_concurrency 2
                for (int ic = 0; ic < in_c_by_word; ic++) {
                  hls_register T_in_hls in_regs[p::max_in_b];

#pragma unroll
#pragma max_concurrency 2
                  for (int ib = 0; ib < p::max_in_b; ib++) { in_regs[ib] = in_buf[ih][iw][ic][ib]; }

#pragma unroll
#pragma max_concurrency 8
                  for (int oc = 0; oc < out_c_low; oc++) {
                    const T_k_hls k_reg = k_buf[kh][kw][ic][oc];
                    const T_out_hls mul_res = PE_kn2row_tiling(k_reg, in_regs);
                    out_regs[oc] += mul_res;
                  }
                }

                if (output_on) {
#pragma unroll
#pragma max_concurrency 8
                  for (int oc = 0; oc < out_c_low; oc++) {
                    const T_out_hls out_pre = out_buf[oh][ow][oc];
                    const bool init_out = ((kh == 0) && (kw == 0));
                    out_buf[oh][ow][oc] = (init_out) ? out_regs[oc] : T_out_hls(out_pre + out_regs[oc]);
                  }
                }
              }
            }
          }
        }

      OUTPUT_STORE_MODULE:
#pragma loop_coalesce 2
#pragma unroll 1
#pragma max_concurrency 1
        for (int oh = 0; oh < p::tile_h; ++oh) {
          for (int ow = 0; ow < p::tile_w; ++ow) {
#pragma unroll
#pragma max_concurrency 8
            for (int oc = 0; oc < out_c_low; oc++) {
              const T_out_hls out = out_buf[oh][ow][oc];
              T_out_hls tmp;

            THRESHOLD_APPLY_UNIT:
              if (use_threshold > 0) {
                const T_out_hls ts0 = threshold_buf[oc][0];
                const T_out_hls ts1 = threshold_buf[oc][1];
                const T_out_hls ts2 = threshold_buf[oc][2];
                const T_out_hls flag = threshold_buf[oc][3];

                if (flag == 1) /// increasing function
                {
                  if (out < ts0)
                    tmp = 0;
                  else if (out < ts1)
                    tmp = 1;
                  else if (out < ts2)
                    tmp = 2;
                  else
                    tmp = 3;
                } else if (flag == -1) /// decreasing function
                {
                  if (out > ts2)
                    tmp = 0;
                  else if (out > ts1)
                    tmp = 1;
                  else if (out > ts0)
                    tmp = 2;
                  else
                    tmp = 3;
                } else {
                  /// max value of 2 bits
                  const T_out_hls k = 3 * 3 * out_c * 3;
                  tmp = flag - k;
                }
              } else {
                tmp = out;
              }

              /// export out data to actual memory space.
              const unsigned oh_ = ih_high + oh;
              const unsigned ow_ = iw_high + ow;
              const unsigned oc_ = oc_high + oc;

              const bool output_on = ((oh_ < out_h) && (ow_ < out_w) && (oc_ < out_c));
              if (output_on) {
                const int _out_w = int(out_w);
                const int _out_c = int(out_c);
                int idx_out = oh_ * _out_w * _out_c + ow_ * _out_c + oc_;
                out_data[idx_out] = T_out_hls(tmp);
              }
            }
          }
        }
      }
    }
  }
}

hls_avalon_slave_component void intel_hls_qconv1x1_optim_impl(
  hls_avalon_slave_register_argument
    ihc::mm_master<T_in_hls, ihc::aspace<1>, ihc::awidth<32>, ihc::dwidth<BW_>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &in_data,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_out_hls, ihc::aspace<2>, ihc::awidth<32>, ihc::dwidth<BW_>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &out_data,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_k_hls, ihc::aspace<3>, ihc::awidth<32>, ihc::dwidth<BW_>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &k_data,
  hls_avalon_slave_register_argument uint32 in_w, hls_avalon_slave_register_argument uint32 in_h,
  hls_avalon_slave_register_argument uint32 in_c_by_word, hls_avalon_slave_register_argument uint32 out_w,
  hls_avalon_slave_register_argument uint32 out_h, hls_avalon_slave_register_argument uint32 out_c)
{
  static const unsigned out_c_low = p::num_pe;

KERNEL_LOAD_MODULE:
#pragma unroll 1
#pragma max_concurrency 1
  for (int oc_high = 0; oc_high < out_c; oc_high += out_c_low) {
    hls_memory hls_singlepump hls_bankbits(0, 1, 2, 3) hls_memory T_k_hls k_buf[p::max_in_c_by_word][out_c_low];

    unsigned idx_in = 0;

#pragma unroll 2
    for (int kc = 0; kc < in_c_by_word; kc++) {
#pragma unroll
      for (int oc = 0; oc < out_c_low; oc++) {
        const unsigned _in_c = unsigned(in_c_by_word);
        const unsigned idx_k = (oc_high / out_c_low) * _in_c * out_c_low + kc * out_c_low + oc;
        k_buf[kc][oc] = k_data[idx_k];
      }
    }

#pragma loop_coalesce 2
#pragma unroll 1
    for (int ih = 0; ih < in_h; ++ih) {
      for (int iw = 0; iw < in_w; ++iw) {
        hls_register T_in_hls out_regs[out_c_low] = {0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll 2
        for (int ic = 0; ic < in_c_by_word; ic++) {
          hls_register T_in_hls in_regs[p::max_in_b];

#pragma unroll
          for (int ib = 0; ib < p::max_in_b; ib++) {
            const unsigned _in_w = unsigned(in_w);
            const unsigned _in_c = unsigned(in_c_by_word);
            // const unsigned idx_in = ih * _in_w * _in_c * p::max_in_b + iw * _in_c * p::max_in_b + ic * p::max_in_b +
            // ib;
            const int idx_k = (kh * _k_w * _in_c * out_c_low) + (kw * _in_c * out_c_low) + (ic * out_c_low) + oc +
                              (oc_high * _k_h * _k_w * _in_c);
            in_regs[ib] = in_data[idx_in++];
          }

#pragma unroll
          for (int oc = 0; oc < out_c_low; oc++) {
            const T_k_hls k_reg = k_buf[ic][oc];
            out_regs[oc] += PE_kn2row_tiling(k_reg, in_regs);
          }
        }

#pragma unroll
        for (int oc = 0; oc < out_c_low; oc++) {
          const unsigned _oc = oc_high + oc;
          const unsigned _out_w = unsigned(in_w);
          const unsigned _out_c = unsigned(out_c);
          const unsigned idx_out = ih * _out_w * _out_c + iw * _out_c + _oc;
          out_data[idx_out] = out_regs[oc];
        }
      }
    }
  }
}

hls_avalon_slave_component void intel_hls_qconv3x3_optim_impl(
  hls_avalon_slave_register_argument
    ihc::mm_master<T_in_hls, ihc::aspace<1>, ihc::awidth<32>, ihc::dwidth<BW_>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &in_data,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_out_hls, ihc::aspace<2>, ihc::awidth<32>, ihc::dwidth<BW_>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &out_data,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_k_hls, ihc::aspace<3>, ihc::awidth<32>, ihc::dwidth<BW_>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &k_data,
  hls_avalon_slave_register_argument uint32 in_w, hls_avalon_slave_register_argument uint32 in_h,
  hls_avalon_slave_register_argument uint32 in_c_by_word, hls_avalon_slave_register_argument uint32 out_w,
  hls_avalon_slave_register_argument uint32 out_h, hls_avalon_slave_register_argument uint32 out_c)
{
  static const unsigned k_h = 3;
  static const unsigned k_w = 3;
  static const unsigned out_c_low = p::num_pe;
  static const unsigned in_h_bank = 4;
  static const unsigned p::in_tile_h_banked = (p::in_tile_h + (in_h_bank - 1)) / in_h_bank;

  for (int oh_high = 0; oh_high < out_c; oh_high += out_c_low) {
    hls_memory hls_singlepump hls_bankbits(0, 1, 2) T_out_hls out_buf[p::tile_h][p::tile_w][out_c_low];

    hls_memory hls_singlepump ac_int<InTypeBitWidth, false> k[out_c_low][k_h][k_w];
    hls_memory hls_singlepump T_out_hls kp[out_c_low];

#pragma unroll 1
    for (int oc = 0; oc < out_c_low; ++oc) {
      int out_ch = oh_high + oc;
      T_out_hls sum = 0;
      // if (out_ch >= out_channels)
      //   break;
#pragma unroll 1
#pragma loop_coalesce 2
      for (int kh = 0; kh < k_h; ++kh) {
#pragma unroll 1
        for (int kw = 0; kw < k_w; ++kw) {
#pragma unroll 1
          for (int kc = 0; kc < in_c_by_word; ++kc) {
            const int oc = oc_high * k_h * k_w * in_c_by_word + kh * k_w + kc;
            ac_int<InTypeBitWidth, false> kernel = kernel_data[];
            k[oc][kr][kc] = kernel;
            sum += (T_out_hls)popcountll(kernel.bit_complement());
          }
        }
      }
      kp[oc] = sum;
    }
    for (int in_bit_ch = 0; in_bit_ch < in_bitwidth; ++in_bit_ch) {
      hls_memory hls_singlepump hls_bankbits(6, 7, 8, 9, 10, 11)
        IN_TYPE lines[TileOutputHeight * 2][TileOutputWidth * 2];
#pragma ivdep array(lines)
#pragma unroll 1
      for (int row = 0; row < TileOutputHeight + 2; ++row) {
        hls_register IN_TYPE v[3][3];
#pragma unroll 1
        for (int col = 0; col < TileOutputWidth + 2; ++col) {
          lines[row][col] = in_data[ic_high * in_bitwidth * in_height * in_width + in_bit_ch * in_height * in_width +
                                    row * in_width + col];
          if (row < 2)
            continue;
#pragma unroll
          for (int kr = 0; kr < 3; ++kr) {
            v[kr][0] = v[kr][1];
            v[kr][1] = v[kr][2];
            v[kr][2] = lines[row - 2 + kr][col];
          }
          if (col < 2)
            continue;
#pragma unroll
          for (int oc = 0; oc < out_c_low; ++oc) {
            int out_ch = oh_high + oc;
            if (out_ch >= out_channels)
              continue;
            ac_int<10, false> xnorsum = 0;
#pragma unroll
            for (int kr = 0; kr < 3; ++kr) {
#pragma unroll
              for (int kc = 0; kc < 3; ++kc) {
                xnorsum += (ac_int<10, false>)popcountll((v[kr][kc] ^ k[oc][kr][kc]).bit_complement());
              }
            }
            T_out_hls tmp = out[row - 2][col - 2][oc];
            T_out_hls add_data = ((int16_t)xnorsum - (int16_t)kp[oc]) << in_bit_ch;
            out[row - 2][col - 2][oc] = (in_bit_ch == 0 && ic_high == 0) ? add_data : tmp + add_data;
          }
        }
      }
    }
#pragma unroll 1
    for (int row = 0; row < TileOutputHeight; ++row) {
#pragma unroll 1
      for (int col = 0; col < TileOutputWidth; ++col) {
#pragma unroll
        for (int oc = 0; oc < out_c_low; ++oc) {
          int out_ch = oh_high + oc;
          if (out_ch >= out_channels)
            continue;
          out_data[row * out_width * out_stride + col * out_stride + out_ch] = out[row][col][oc];
        }
      }
    }
  }
}