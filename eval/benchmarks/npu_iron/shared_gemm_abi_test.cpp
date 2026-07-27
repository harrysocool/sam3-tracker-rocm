// Validate six dynamic-K GEMM instruction streams against one common xclbin.
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <stdexcept>
#include <vector>

using std::string;
using std::vector;
using bf16 = uint16_t;

struct Shape {
  const char *name;
  const char *artifact;
  int m;
  int k;
  int n;
  bool compact;
};

static vector<uint8_t> read_binary(const string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("cannot open " + path);
  const size_t size = static_cast<size_t>(f.tellg());
  f.seekg(0);
  vector<uint8_t> data(size);
  f.read(reinterpret_cast<char *>(data.data()), size);
  return data;
}

int main(int argc, char **argv) {
  constexpr bf16 one_bf16 = 0x3f80;
  constexpr float tolerance = 4.0f;
  string xclbin_shape = "qkv_w";
  string only_shape;
  string root =
      "/home/amd/project/npu_iron/sam3_attn/"
      "shared_gemm_dynamic_rtp_v5/";
  string compact_root =
      "/home/amd/project/npu_iron/sam3_attn/"
      "compact_ffn_dynamic_rtp_v5/";
  for (int i = 1; i < argc; ++i) {
    const string arg = argv[i];
    if (arg == "--xclbin-shape" && i + 1 < argc)
      xclbin_shape = argv[++i];
    else if (arg == "--only-shape" && i + 1 < argc)
      only_shape = argv[++i];
    else if (arg == "--artifact-root" && i + 1 < argc)
      root = argv[++i];
    else if (arg == "--compact-root" && i + 1 < argc)
      compact_root = argv[++i];
    else {
      std::fprintf(stderr, "unknown or incomplete argument: %s\n", argv[i]);
      return 2;
    }
  }
  if (root.empty() || root.back() != '/')
    root.push_back('/');
  if (compact_root.empty() || compact_root.back() != '/')
    compact_root.push_back('/');

  const Shape shapes[] = {
      {"o_g", "o_g", 1536, 1024, 1024, false},
      {"o_w", "o_w", 2304, 1024, 1024, false},
      {"qkv_g", "qkv_g", 1536, 1024, 3072, false},
      {"qkv_w", "qkv_w", 2304, 1024, 3072, false},
      {"ffn1", "ffn1", 1536, 1024, 5120, false},
      {"ffn2", "ffn2", 1536, 5120, 1024, false},
      // Exercise both K=4 -> K=20 and K=20 -> K=4 boundaries, then repeat
      // K=20. Also execute compact FFN1 N=4864 through the common xclbin.
      {"o_g_repeat", "o_g", 1536, 1024, 1024, false},
      {"ffn1_compact", "ffn1", 1536, 1024, 4864, true},
      {"ffn2_repeat", "ffn2", 1536, 5120, 1024, false},
  };

  xrt::device device(0);
  xrt::xclbin xclbin(root + xclbin_shape + "/final.xclbin");
  const auto uuid = device.register_xclbin(xclbin);
  xrt::hw_context context(device, uuid);
  xrt::kernel kernel(context, "MLIR_AIE");

  size_t executed = 0;
  for (const auto &shape : shapes) {
    if (!only_shape.empty() && only_shape != shape.name)
      continue;
    ++executed;
    std::printf("shared_abi start shape=%s M=%d K=%d N=%d\n", shape.name,
                shape.m, shape.k, shape.n);
    std::fflush(stdout);

    const string &shape_root = shape.compact ? compact_root : root;
    const auto inst = read_binary(shape_root + shape.artifact + "/insts.bin");
    xrt::bo inst_bo(device, inst.size(), xrt::bo::flags::cacheable,
                    kernel.group_id(1));
    inst_bo.write(inst.data());
    inst_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    const size_t a_elems = static_cast<size_t>(shape.m) * shape.k;
    const size_t b_elems = static_cast<size_t>(shape.k) * shape.n;
    const size_t c_elems = static_cast<size_t>(shape.m) * shape.n;
    xrt::bo a_bo(device, a_elems * sizeof(bf16), xrt::bo::flags::host_only,
                 kernel.group_id(3));
    xrt::bo b_bo(device, b_elems * sizeof(bf16), xrt::bo::flags::host_only,
                 kernel.group_id(4));
    xrt::bo c_bo(device, c_elems * sizeof(float), xrt::bo::flags::host_only,
                 kernel.group_id(5));

    vector<bf16> a(a_elems, one_bf16);
    vector<bf16> b(b_elems, one_bf16);
    a_bo.write(a.data());
    b_bo.write(b.data());
    a_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    b_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = kernel(3, inst_bo, static_cast<uint32_t>(inst.size() / 4),
                      a_bo, b_bo, c_bo);
    run.wait();
    c_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const float *out = c_bo.map<const float *>();
    const float expected = static_cast<float>(shape.k);

    double sum = 0.0;
    float max_abs = 0.0f;
    size_t bad = 0;
    size_t first_bad = c_elems;
    size_t last_bad = 0;
    for (size_t i = 0; i < c_elems; ++i) {
      const float value = out[i];
      const float error = std::fabs(value - expected);
      sum += value;
      max_abs = std::max(max_abs, error);
      if (!std::isfinite(value) || error > tolerance) {
        first_bad = std::min(first_bad, i);
        last_bad = i;
        ++bad;
      }
    }
    const double mean = sum / static_cast<double>(c_elems);
    std::printf(
        "shared_abi result shape=%s mean=%.6f max_abs=%.6f bad=%zu/%zu\n",
        shape.name, mean, max_abs, bad, c_elems);
    std::fflush(stdout);
    if (bad != 0) {
      std::printf(
          "shared_abi bad_range shape=%s first_index=%zu first_row=%zu "
          "last_index=%zu last_row=%zu first_value=%.6f\n",
          shape.name, first_bad, first_bad / shape.n, last_bad,
          last_bad / shape.n, out[first_bad]);
      std::fflush(stdout);
    }
    if (bad != 0)
      return 1;
  }

  if (executed == 0) {
    std::fprintf(stderr, "no matching shape selected\n");
    return 2;
  }
  std::puts("SHARED_GEMM_ABI_PASS");
  return 0;
}
