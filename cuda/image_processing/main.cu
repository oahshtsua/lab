#include <cmath>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <texture_types.h>
#include <time.h>

#define W_IN 1920
#define H_IN 1080
#define W_OUT 3840
#define H_OUT 2160
#define MASK_SIZE 19
#define OUTPUT_SIZE (W_OUT * H_OUT)

#define MALLOC_CHECK(ptr)                                                      \
  {                                                                            \
    if (!ptr) {                                                                \
      fprintf(stderr, "Error allocating CPU memory at %s: %d\n", __FILE__,     \
              __LINE__);                                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CUDA_CHECK(err)                                                        \
  {                                                                            \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "Cuda runtime error at %s: %d: %s\n", __FILE__,          \
              __LINE__, cudaGetErrorString(err));                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

__constant__ float c_mask[MASK_SIZE * MASK_SIZE];

void init_mask(float *mask);
void cpu_upscale_bilinear(const float *img_in, float *img_out);
void cpu_convolve_zero_padded(const float *src, size_t src_w, size_t src_h,
                              const float *mask, size_t mask_w, size_t mask_h,
                              float *dest);

void verify_correctness(float *ref, float *val, size_t n);

__global__ void gpu_upscale_bilinear_global(const float *src, float *dst,
                                            int w_in, int h_in, int w_out,
                                            int h_out);
__global__ void gpu_convolve_global(const float *src, const float *mask,
                                    float *dst, int img_w, int img_h,
                                    int mask_w, int mask_h);
__global__ void gpu_convolve_constant(const float *src, float *dst, int img_w,
                                      int img_h, int mask_w, int mask_h);
__global__ void gpu_upscale_bilinear_texture(float *dst, int w_in, int h_in,
                                             int w_out, int h_out,
                                             cudaTextureObject_t texObj);
__global__ void gpu_convolve_texture(float *dst, int img_w, int img_h,
                                     int mask_w, int mask_h,
                                     cudaTextureObject_t texObj);
void log_results(const char *filename, const char *benchmark_name,
                 float upscaling_time_ms, float convolution_time_ms);

void bmark_gpu_global(float *d_in, float *d_up, float *d_out, float *d_mask);
void bmark_gpu_constant(float *d_in, float *d_up, float *d_out, float *h_mask);
void bmark_gpu_texture(float *d_in, float *d_up, float *d_out, float *h_mask);

int main() {
  const size_t input_size = W_IN * H_IN;
  const size_t output_size = W_OUT * H_OUT;

  float mask[MASK_SIZE * MASK_SIZE];
  float *img_in = (float *)malloc(input_size * sizeof(float));
  MALLOC_CHECK(img_in);
  float *img_out = (float *)malloc(output_size * sizeof(float));
  MALLOC_CHECK(img_out);
  float *img_conv = (float *)malloc(output_size * sizeof(float));
  MALLOC_CHECK(img_conv);
  float *gpu_res = (float *)malloc(output_size * sizeof(float));
  MALLOC_CHECK(gpu_res);

  // Load input image (single channel)
  FILE *f_in = fopen("input.raw", "rb");
  if (!f_in) {
    fprintf(stderr, "ERROR: Could not open input.raw\n");
    return 1;
  }
  size_t items = fread(img_in, sizeof(float), input_size, f_in);
  fclose(f_in);
  if (items != input_size) {
    fprintf(stderr, "ERROR: Input file size is incorrect!\n");
    return 1;
  }
  printf("Loaded input.raw (%zu floats)\n", items);

  // Initialize mask
  init_mask(mask);

  // CPU pipeline
  clock_t t0 = clock();
  cpu_upscale_bilinear(img_in, img_out);
  clock_t t1 = clock();
  float cpu_upscale_ms = (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0;
  printf("CPU upscaling done in %.3f ms.\n", cpu_upscale_ms);

  clock_t t2 = clock();
  cpu_convolve_zero_padded(img_out, W_OUT, H_OUT, mask, MASK_SIZE, MASK_SIZE,
                           img_conv);
  clock_t t3 = clock();
  float cpu_convolve_ms = (double)(t3 - t2) / CLOCKS_PER_SEC * 1000.0;
  printf("CPU convolution done in %.3f ms.\n", cpu_convolve_ms);

  log_results("metrics.csv", "CPU_Total", cpu_upscale_ms, cpu_convolve_ms);

  // Save CPU result
  FILE *f_out = fopen("output_conv.raw", "wb");
  if (!f_out) {
    fprintf(stderr, "ERROR: Could not open output_conv.raw for writing.\n");
    return 1;
  }
  fwrite(img_conv, sizeof(float), output_size, f_out);
  fclose(f_out);
  printf("Saved output_conv.raw (%zu floats)\n", output_size);

  // GPU memory allocation
  float *d_in = NULL, *d_up = NULL, *d_out = NULL, *d_mask = NULL;
  CUDA_CHECK(cudaMalloc(&d_in, input_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_up, output_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, output_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_mask, MASK_SIZE * MASK_SIZE * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_in, img_in, input_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_mask, mask, MASK_SIZE * MASK_SIZE * sizeof(float),
                        cudaMemcpyHostToDevice));

  // GPU benchmark: global memory mask
  printf("\n--- GPU benchmark (global memory mask) ---\n");
  bmark_gpu_global(d_in, d_up, d_out, d_mask);

  CUDA_CHECK(cudaMemcpy(gpu_res, d_out, output_size * sizeof(float),
                        cudaMemcpyDeviceToHost));
  verify_correctness(img_conv, gpu_res, output_size);

  FILE *f_gpu_global = fopen("output_gpu_global.raw", "wb");
  fwrite(gpu_res, sizeof(float), output_size, f_gpu_global);
  fclose(f_gpu_global);
  printf("Saved output_gpu_global.raw (%zu floats)\n", output_size);

  // GPU benchmark: constant memory mask
  printf("\n--- GPU benchmark (constant memory mask) ---\n");
  bmark_gpu_constant(d_in, d_up, d_out, mask);

  CUDA_CHECK(cudaMemcpy(gpu_res, d_out, output_size * sizeof(float),
                        cudaMemcpyDeviceToHost));
  verify_correctness(img_conv, gpu_res, output_size);
  FILE *f_gpu_const = fopen("output_gpu_const.raw", "wb");
  fwrite(gpu_res, sizeof(float), output_size, f_gpu_const);
  fclose(f_gpu_const);
  printf("Saved output_gpu_const.raw (%zu floats)\n", output_size);

  // GPU benchmark: texture memory
  printf("\n--- GPU benchmark (texture memory) ---\n");
  bmark_gpu_texture(d_in, d_up, d_out, mask);

  CUDA_CHECK(cudaMemcpy(gpu_res, d_out, output_size * sizeof(float),
                        cudaMemcpyDeviceToHost));
  verify_correctness(img_conv, gpu_res, output_size);
  FILE *f_gpu_tex = fopen("output_gpu_texture.raw", "wb");
  fwrite(gpu_res, sizeof(float), output_size, f_gpu_tex);
  fclose(f_gpu_tex);
  printf("Saved output_gpu_texture.raw (%zu floats)\n", output_size);

  // Cleanup
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_up));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_mask));

  free(img_in);
  free(img_out);
  free(img_conv);
  free(gpu_res);

  return 0;
}

void log_results(const char *filename, const char *benchmark_name,
                 float upscaling_time_ms, float convolution_time_ms) {
  FILE *fp = fopen(filename, "a");
  if (!fp) {
    fprintf(stderr, "Error opening log file: %s\n", filename);
    exit(EXIT_FAILURE);
  }
  fprintf(fp, "%s,%.3f,%.3f\n", benchmark_name, upscaling_time_ms,
          convolution_time_ms);
  fclose(fp);
} /* log_results */

void init_mask(float *mask) {
  float val = 1.0f / (MASK_SIZE * MASK_SIZE);
  for (int i = 0; i < MASK_SIZE * MASK_SIZE; i++)
    mask[i] = val;
} /* init_mask */

void cpu_upscale_bilinear(const float *img_in, float *img_out) {
  const float scale_x = (float)W_IN / W_OUT;
  const float scale_y = (float)H_IN / H_OUT;

  for (size_t y_out = 0; y_out < H_OUT; y_out++) {
    for (size_t x_out = 0; x_out < W_OUT; x_out++) {
      float x = (x_out + 0.5f) * scale_x - 0.5f;
      float y = (y_out + 0.5f) * scale_y - 0.5f;

      int x1 = (int)floorf(x);
      int y1 = (int)floorf(y);
      int x2 = x1 + 1;
      int y2 = y1 + 1;

      float tx = x - x1;
      float ty = y - y1;

      x1 = (x1 < 0) ? 0 : x1;
      y1 = (y1 < 0) ? 0 : y1;
      x2 = (x2 >= W_IN) ? W_IN - 1 : x2;
      y2 = (y2 >= H_IN) ? H_IN - 1 : y2;

      float Q11 = img_in[y1 * W_IN + x1];
      float Q21 = img_in[y1 * W_IN + x2];
      float Q12 = img_in[y2 * W_IN + x1];
      float Q22 = img_in[y2 * W_IN + x2];

      float R1 = Q11 * (1.0f - tx) + Q21 * tx;
      float R2 = Q12 * (1.0f - tx) + Q22 * tx;
      float P = R1 * (1.0f - ty) + R2 * ty;

      img_out[y_out * W_OUT + x_out] = P;
    }
  }
} /* cpu_upscale_bilinear */

void cpu_convolve_zero_padded(const float *src, size_t src_w, size_t src_h,
                              const float *mask, size_t mask_w, size_t mask_h,
                              float *dest) {
  size_t mask_center_x = mask_w / 2;
  size_t mask_center_y = mask_h / 2;

  for (size_t y = 0; y < src_h; y++) {
    for (size_t x = 0; x < src_w; x++) {
      float sum = 0.0;
      for (size_t my = 0; my < mask_h; my++) {
        int sy = y + (my - mask_center_y);
        if (sy < 0 || sy >= src_h)
          continue;
        for (size_t mx = 0; mx < mask_w; mx++) {
          int sx = x + (mx - mask_center_x);
          if (sx < 0 || sx >= src_w)
            continue;
          sum += src[sy * src_w + sx] * mask[my * mask_w + mx];
        }
      }
      dest[y * src_w + x] = sum;
    }
  }
} /* cpu_convolve_zero_padded */

__global__ void gpu_upscale_bilinear_global(const float *src, float *dst,
                                            int w_in, int h_in, int w_out,
                                            int h_out) {
  int x_out = blockIdx.x * blockDim.x + threadIdx.x;
  int y_out = blockIdx.y * blockDim.y + threadIdx.y;

  if (x_out >= w_out || y_out >= h_out)
    return;

  float scale_x = (float)w_in / w_out;
  float scale_y = (float)h_in / h_out;

  float x = (x_out + 0.5f) * scale_x - 0.5f;
  float y = (y_out + 0.5f) * scale_y - 0.5f;

  int x1 = floorf(x);
  int y1 = floorf(y);
  int x2 = x1 + 1;
  int y2 = y1 + 1;

  float tx = x - x1;
  float ty = y - y1;

  x1 = (x1 < 0) ? 0 : x1;
  y1 = (y1 < 0) ? 0 : y1;
  x2 = (x2 >= w_in) ? w_in - 1 : x2;
  y2 = (y2 >= h_in) ? h_in - 1 : y2;

  float Q11 = src[y1 * w_in + x1];
  float Q21 = src[y1 * w_in + x2];
  float Q12 = src[y2 * w_in + x1];
  float Q22 = src[y2 * w_in + x2];

  float R1 = Q11 * (1.0f - tx) + Q21 * tx;
  float R2 = Q12 * (1.0f - tx) + Q22 * tx;
  float P = R1 * (1.0f - ty) + R2 * ty;

  dst[y_out * w_out + x_out] = P;
} /* gpu_upscale_bilinear_global */

__global__ void gpu_convolve_global(const float *src, const float *mask,
                                    float *dst, int img_w, int img_h,
                                    int mask_w, int mask_h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= img_w || y >= img_h)
    return;

  int mask_cx = mask_w / 2;
  int mask_cy = mask_h / 2;
  float sum = 0.0f;

  for (int my = 0; my < mask_h; my++) {
    int sy = y + (my - mask_cy);
    if (sy < 0 || sy >= img_h)
      continue;
    for (int mx = 0; mx < mask_w; mx++) {
      int sx = x + (mx - mask_cx);
      if (sx < 0 || sx >= img_w)
        continue;
      sum += src[sy * img_w + sx] * mask[my * mask_w + mx];
    }
  }
  dst[y * img_w + x] = sum;
} /* gpu_convolve_global */

__global__ void gpu_convolve_constant(const float *src, float *dst, int img_w,
                                      int img_h, int mask_w, int mask_h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= img_w || y >= img_h)
    return;

  int mask_cx = mask_w / 2;
  int mask_cy = mask_h / 2;
  float sum = 0.0f;

  for (int my = 0; my < mask_h; my++) {
    int sy = y + (my - mask_cy);
    if (sy < 0 || sy >= img_h)
      continue;
    for (int mx = 0; mx < mask_w; mx++) {
      int sx = x + (mx - mask_cx);
      if (sx < 0 || sx >= img_w)
        continue;
      sum += src[sy * img_w + sx] * c_mask[my * mask_w + mx];
    }
  }
  dst[y * img_w + x] = sum;
} /* gpu_convolve_constant */

__global__ void gpu_upscale_bilinear_texture(float *dst, int w_in, int h_in,
                                             int w_out, int h_out,
                                             cudaTextureObject_t texObj) {
  int x_out = blockIdx.x * blockDim.x + threadIdx.x;
  int y_out = blockIdx.y * blockDim.y + threadIdx.y;

  if (x_out >= w_out || y_out >= h_out)
    return;

  float scale_x = (float)w_in / w_out;
  float scale_y = (float)h_in / h_out;

  float x = (x_out + 0.5f) * scale_x - 0.5f;
  float y = (y_out + 0.5f) * scale_y - 0.5f;

  float val = tex2D<float>(texObj, x, y);

  dst[y_out * w_out + x_out] = val;
} /* gpu_upscale_bilinear_texture */

__global__ void gpu_convolve_texture(float *dst, int img_w, int img_h,
                                     int mask_w, int mask_h,
                                     cudaTextureObject_t texObj) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= img_w || y >= img_h)
    return;

  int cx = mask_w / 2;
  int cy = mask_h / 2;
  float sum = 0.0f;

  for (int my = 0; my < mask_h; my++) {
    for (int mx = 0; mx < mask_w; mx++) {
      int xx = x + (mx - cx);
      int yy = y + (my - cy);

      float pixel = tex2D<float>(texObj, (float)xx, (float)yy);

      sum += pixel * c_mask[my * mask_w + mx];
    }
  }

  dst[y * img_w + x] = sum;
} /* gpu_convolve_texture */

void bmark_gpu_global(float *d_in, float *d_up, float *d_out, float *d_mask) {
  dim3 block(16, 16);
  dim3 grid((W_OUT + block.x - 1) / block.x, (H_OUT + block.y - 1) / block.y);

  cudaEvent_t start, stop;
  float upscale_ms, convolve_ms;

  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // warmup run
  gpu_upscale_bilinear_global<<<grid, block>>>(d_in, d_up, W_IN, H_IN, W_OUT,
                                               H_OUT);
  gpu_convolve_global<<<grid, block>>>(d_up, d_mask, d_out, W_OUT, H_OUT,
                                       MASK_SIZE, MASK_SIZE);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Upscaling
  CUDA_CHECK(cudaEventRecord(start));
  gpu_upscale_bilinear_global<<<grid, block>>>(d_in, d_up, W_IN, H_IN, W_OUT,
                                               H_OUT);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&upscale_ms, start, stop));
  printf("GPU upscaling time: %.3f ms\n", upscale_ms);

  // Convolution
  CUDA_CHECK(cudaEventRecord(start));
  gpu_convolve_global<<<grid, block>>>(d_up, d_mask, d_out, W_OUT, H_OUT,
                                       MASK_SIZE, MASK_SIZE);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&convolve_ms, start, stop));
  printf("GPU convolution time: %.3f ms\n", convolve_ms);

  log_results("metrics.csv", "GPU_Global", upscale_ms, convolve_ms);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
} /* bmark_gpu_global */

void bmark_gpu_constant(float *d_in, float *d_up, float *d_out, float *h_mask) {
  // Copy mask to constant memory
  CUDA_CHECK(cudaMemcpyToSymbol(c_mask, h_mask,
                                MASK_SIZE * MASK_SIZE * sizeof(float)));

  dim3 block(16, 16);
  dim3 grid((W_OUT + block.x - 1) / block.x, (H_OUT + block.y - 1) / block.y);

  cudaEvent_t start, stop;
  float upscale_ms, convolve_ms;

  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Warmup run
  gpu_upscale_bilinear_global<<<grid, block>>>(d_in, d_up, W_IN, H_IN, W_OUT,
                                               H_OUT);
  gpu_convolve_constant<<<grid, block>>>(d_up, d_out, W_OUT, H_OUT, MASK_SIZE,
                                         MASK_SIZE);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Upscaling
  CUDA_CHECK(cudaEventRecord(start));
  gpu_upscale_bilinear_global<<<grid, block>>>(d_in, d_up, W_IN, H_IN, W_OUT,
                                               H_OUT);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&upscale_ms, start, stop));
  printf("GPU upscaling (const mask) time: %.3f ms\n", upscale_ms);

  // Convolution
  CUDA_CHECK(cudaEventRecord(start));
  gpu_convolve_constant<<<grid, block>>>(d_up, d_out, W_OUT, H_OUT, MASK_SIZE,
                                         MASK_SIZE);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&convolve_ms, start, stop));
  printf("GPU convolution (const mask) time: %.3f ms\n", convolve_ms);

  log_results("metrics.csv", "GPU_Constant", upscale_ms, convolve_ms);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
} /* bmark_gpu_constant */

void bmark_gpu_texture(float *d_in, float *d_up, float *d_out, float *h_mask) {
  // Copy mask to constant memory
  CUDA_CHECK(cudaMemcpyToSymbol(c_mask, h_mask,
                                MASK_SIZE * MASK_SIZE * sizeof(float)));

  cudaTextureObject_t texObj;
  cudaResourceDesc resDesc = {};
  resDesc.resType = cudaResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = d_in;
  resDesc.res.pitch2D.width = W_IN;
  resDesc.res.pitch2D.height = H_IN;
  resDesc.res.pitch2D.pitchInBytes = W_IN * sizeof(float);
  resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();

  cudaTextureDesc texDesc = {};
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;
  texDesc.filterMode = cudaFilterModeLinear; // bilinear interpolation
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;

  CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

  // Create texture for the upscaled image (for convolution) - use BORDER so
  // convolution sees zeros outside the image (zero-padding).
  cudaTextureObject_t texObj_conv;
  cudaResourceDesc resDesc_conv = {};
  resDesc_conv.resType = cudaResourceTypePitch2D;
  resDesc_conv.res.pitch2D.devPtr = d_up;
  resDesc_conv.res.pitch2D.width = W_OUT;
  resDesc_conv.res.pitch2D.height = H_OUT;
  resDesc_conv.res.pitch2D.pitchInBytes = W_OUT * sizeof(float);
  resDesc_conv.res.pitch2D.desc = cudaCreateChannelDesc<float>();

  cudaTextureDesc texDesc_conv = {};
  texDesc_conv.readMode = cudaReadModeElementType;
  texDesc_conv.normalizedCoords = 0;
  texDesc_conv.filterMode =
      cudaFilterModePoint; // point sampling for convolution
  texDesc_conv.addressMode[0] = cudaAddressModeBorder;
  texDesc_conv.addressMode[1] = cudaAddressModeBorder;

  CUDA_CHECK(cudaCreateTextureObject(&texObj_conv, &resDesc_conv, &texDesc_conv,
                                     nullptr));

  dim3 block(16, 16);
  dim3 grid((W_OUT + block.x - 1) / block.x, (H_OUT + block.y - 1) / block.y);

  cudaEvent_t start, stop;
  float upscale_ms, convolve_ms;

  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Warmup
  gpu_upscale_bilinear_texture<<<grid, block>>>(d_up, W_IN, H_IN, W_OUT, H_OUT,
                                                texObj);
  CUDA_CHECK(cudaDeviceSynchronize());
  gpu_convolve_texture<<<grid, block>>>(d_out, W_OUT, H_OUT, MASK_SIZE,
                                        MASK_SIZE, texObj_conv);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Upscaling
  CUDA_CHECK(cudaEventRecord(start));
  gpu_upscale_bilinear_texture<<<grid, block>>>(d_up, W_IN, H_IN, W_OUT, H_OUT,
                                                texObj);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&upscale_ms, start, stop));
  printf("GPU upscaling (texture) time: %.3f ms\n", upscale_ms);

  // Convolution
  CUDA_CHECK(cudaEventRecord(start));
  gpu_convolve_texture<<<grid, block>>>(d_out, W_OUT, H_OUT, MASK_SIZE,
                                        MASK_SIZE, texObj_conv);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&convolve_ms, start, stop));
  printf("GPU convolution (texture) time: %.3f ms\n", convolve_ms);

  log_results("metrics.csv", "GPU_Texture", upscale_ms, convolve_ms);

  CUDA_CHECK(cudaDestroyTextureObject(texObj_conv));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaDestroyTextureObject(texObj));
} /* bmark_gpu_texture */

void verify_correctness(float *ref, float *val, size_t n) {
  const double error_margin = 1e-3;
  for (size_t i = 0; i < n; i++) {
    if (fabs(ref[i] - val[i]) > error_margin) {
      if (fabs(ref[i] - val[i]) > 1.0) {
        printf("Results dont match at all! %ld, %f, %f\n", i, ref[i], val[i]);
        exit(EXIT_FAILURE);
      }
      printf("Results dont satisfy error threshold!\n");
    }
  }
} /* verify_correctness */
