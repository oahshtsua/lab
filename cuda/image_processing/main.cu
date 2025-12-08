
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define W_IN 1920
#define H_IN 1080
#define W_OUT 3840
#define H_OUT 2160
#define MASK_SIZE 19

__constant__ float c_mask[MASK_SIZE * MASK_SIZE];

void init_mask(float *mask);
void cpu_upscale_bilinear(const float *img_in, float *img_out);
void cpu_convolve_zero_padded(const float *src, size_t src_w, size_t src_h,
                              const float *mask, size_t mask_w, size_t mask_h,
                              float *dest);
__global__ void gpu_upscale_bilinear_global(const float *src, float *dst,
                                            int w_in, int h_in, int w_out,
                                            int h_out);
__global__ void gpu_convolve_global(const float *src, const float *mask,
                                    float *dst, int img_w, int img_h,
                                    int mask_w, int mask_h);
__global__ void gpu_convolve_constant(const float *src, float *dst, int img_w,
                                      int img_h, int mask_w, int mask_h);

void bmark_gpu_global(float *d_in, float *d_up, float *d_out, float *d_mask);
void bmark_gpu_constant(float *d_in, float *d_up, float *d_out, float *h_mask);

int main() {
  const size_t input_size = W_IN * H_IN;
  const size_t output_size = W_OUT * H_OUT;

  float mask[MASK_SIZE * MASK_SIZE];
  float *img_in = (float *)malloc(input_size * sizeof(float));
  float *img_out = (float *)malloc(output_size * sizeof(float));
  float *img_conv = (float *)malloc(output_size * sizeof(float));

  if (!img_in || !img_out || !img_conv) {
    fprintf(stderr, "ERROR: Out of memory.\n");
    return 1;
  }

  // --- Load input image ---
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

  // --- Initialize mask ---
  init_mask(mask);

  // --- CPU pipeline ---
  clock_t t0 = clock();
  cpu_upscale_bilinear(img_in, img_out);
  clock_t t1 = clock();
  printf("CPU upscaling done in %.3f ms.\n",
         (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0);

  clock_t t2 = clock();
  cpu_convolve_zero_padded(img_out, W_OUT, H_OUT, mask, MASK_SIZE, MASK_SIZE,
                           img_conv);
  clock_t t3 = clock();
  printf("CPU convolution done in %.3f ms.\n",
         (double)(t3 - t2) / CLOCKS_PER_SEC * 1000.0);

  // --- Save CPU result ---
  FILE *f_out = fopen("output_conv.raw", "wb");
  if (!f_out) {
    fprintf(stderr, "ERROR: Could not open output_conv.raw for writing.\n");
    return 1;
  }
  fwrite(img_conv, sizeof(float), output_size, f_out);
  fclose(f_out);
  printf("Saved output_conv.raw (%zu floats)\n", output_size);

  // --- GPU memory allocation ---
  float *d_in = NULL, *d_up = NULL, *d_out = NULL, *d_mask = NULL;
  cudaMalloc(&d_in, input_size * sizeof(float));
  cudaMalloc(&d_up, output_size * sizeof(float));
  cudaMalloc(&d_out, output_size * sizeof(float));
  cudaMalloc(&d_mask, MASK_SIZE * MASK_SIZE * sizeof(float));

  cudaMemcpy(d_in, img_in, input_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask, mask, MASK_SIZE * MASK_SIZE * sizeof(float),
             cudaMemcpyHostToDevice);

  // --- GPU benchmark: global memory mask ---
  printf("\n--- GPU benchmark (global memory mask) ---\n");
  bmark_gpu_global(d_in, d_up, d_out, d_mask);

  cudaMemcpy(img_conv, d_out, output_size * sizeof(float),
             cudaMemcpyDeviceToHost);
  FILE *f_gpu_global = fopen("output_gpu_global.raw", "wb");
  fwrite(img_conv, sizeof(float), output_size, f_gpu_global);
  fclose(f_gpu_global);
  printf("Saved output_gpu_global.raw (%zu floats)\n", output_size);

  // --- GPU benchmark: constant memory mask ---
  printf("\n--- GPU benchmark (constant memory mask) ---\n");
  bmark_gpu_constant(d_in, d_up, d_out, mask);

  cudaMemcpy(img_conv, d_out, output_size * sizeof(float),
             cudaMemcpyDeviceToHost);
  FILE *f_gpu_const = fopen("output_gpu_const.raw", "wb");
  fwrite(img_conv, sizeof(float), output_size, f_gpu_const);
  fclose(f_gpu_const);
  printf("Saved output_gpu_const.raw (%zu floats)\n", output_size);

  // --- Free GPU memory ---
  cudaFree(d_in);
  cudaFree(d_up);
  cudaFree(d_out);
  cudaFree(d_mask);

  // --- Free CPU memory ---
  free(img_in);
  free(img_out);
  free(img_conv);

  return 0;
}

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

      if (x1 < 0)
        x1 = 0;
      if (y1 < 0)
        y1 = 0;
      if (x2 >= W_IN)
        x2 = W_IN - 1;
      if (y2 >= H_IN)
        y2 = H_IN - 1;

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

__global__ void gpu_convolve_global(const float *src, size_t src_w,
                                    size_t src_h, const float *mask,
                                    size_t mask_w, size_t mask_h, float *dest) {

  int x = blockIdx.x * blockDim.x * threadIdx.x;
  int y = blockIdx.y * blockDim.y * threadIdx.y;
  if (x >= src_w || y >= src_h)
    return;

  size_t mask_center_x = mask_w / 2;
  size_t mask_center_y = mask_h / 2;

  float sum = 0.0f;
  for (int my = 0; my < mask_h; my++) {
    int sy = y + (my - mask_center_y);
    if (sy < 0 || sy >= src_h)
      continue;
    for (int mx = 0; mx < mask_w; mx++) {
      int sx = x + (mx - mask_center_x);
      if (sx < 0 || sx >= src_w)
        continue;
      sum += src[sy * src_w + sx] * mask[my * mask_w + mx];
    }
  }
  dest[y * src_w + x] = sum;
} /* gpu_convolve_global */

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

  if (x1 < 0)
    x1 = 0;
  if (y1 < 0)
    y1 = 0;
  if (x2 >= w_in)
    x2 = w_in - 1;
  if (y2 >= h_in)
    y2 = h_in - 1;

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
}

void bmark_gpu_global(float *d_in, float *d_up, float *d_out, float *d_mask) {
  dim3 block(16, 16);
  dim3 grid((W_OUT + block.x - 1) / block.x, (H_OUT + block.y - 1) / block.y);

  cudaEvent_t start, stop;
  float ms;

  // warmup run
  gpu_upscale_bilinear_global<<<grid, block>>>(d_in, d_up, W_IN, H_IN, W_OUT,
                                               H_OUT);
  gpu_convolve_global<<<grid, block>>>(d_up, d_mask, d_out, W_OUT, H_OUT,
                                       MASK_SIZE, MASK_SIZE);
  cudaDeviceSynchronize();

  // --- Upscaling ---
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  gpu_upscale_bilinear_global<<<grid, block>>>(d_in, d_up, W_IN, H_IN, W_OUT,
                                               H_OUT);
  cudaDeviceSynchronize();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  printf("GPU upscaling time: %.3f ms\n", ms);

  // --- Convolution ---
  cudaEventRecord(start);

  gpu_convolve_global<<<grid, block>>>(d_up, d_mask, d_out, W_OUT, H_OUT,
                                       MASK_SIZE, MASK_SIZE);
  cudaDeviceSynchronize();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  printf("GPU convolution time: %.3f ms\n", ms);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
} /* bmark_gpu_global */

void bmark_gpu_constant(float *d_in, float *d_up, float *d_out, float *h_mask) {
  // Copy mask to constant memory
  cudaMemcpyToSymbol(c_mask, h_mask, MASK_SIZE * MASK_SIZE * sizeof(float));

  dim3 block(16, 16);
  dim3 grid((W_OUT + block.x - 1) / block.x, (H_OUT + block.y - 1) / block.y);

  cudaEvent_t start, stop;
  float ms;

  // --- Warmup run ---
  gpu_upscale_bilinear_global<<<grid, block>>>(d_in, d_up, W_IN, H_IN, W_OUT,
                                               H_OUT);
  gpu_convolve_constant<<<grid, block>>>(d_up, d_out, W_OUT, H_OUT, MASK_SIZE,
                                         MASK_SIZE);
  cudaDeviceSynchronize();

  // --- Timed upscaling ---
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  gpu_upscale_bilinear_global<<<grid, block>>>(d_in, d_up, W_IN, H_IN, W_OUT,
                                               H_OUT);
  cudaDeviceSynchronize();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  printf("GPU upscaling (const mask) time: %.3f ms\n", ms);

  // --- Timed convolution ---
  cudaEventRecord(start);
  gpu_convolve_constant<<<grid, block>>>(d_up, d_out, W_OUT, H_OUT, MASK_SIZE,
                                         MASK_SIZE);
  cudaDeviceSynchronize();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  printf("GPU convolution (const mask) time: %.3f ms\n", ms);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
} /* bmark_gpu_constant */
