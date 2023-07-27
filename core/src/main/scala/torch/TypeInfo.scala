package torch

// ffinfo is not available in libtorch, because C++ it's possible to read these values directly
// TODO add something to preset to read values directly
// https://github.com/pytorch/pytorch/blob/377f306b4c22b569588b7d56b8d394a6f985df2f/torch/csrc/TypeInfo.cpp#L297
// https://discuss.pytorch.org/t/how-to-get-finfo-in-c-torchlib-like-that-in-pytorch/131787

case class FInfo(
    resolution: Double,
    min: Double,
    max: Double,
    eps: Double,
    smallest_normal: Double,
    tiny: Double,
    dtype: String
)


def finfo[D <: FloatNN | ComplexNN](dtype: D): FInfo =
  dtype match
    case _: Float16 | _: Complex32 =>
      FInfo(
        resolution = 0.001,
        min = -65504,
        max = 65504,
        eps = 0.000976562,
        smallest_normal = 6.10352e-05,
        tiny = 6.10352e-05,
        dtype = "float16"
      )
    case d: BFloat16 =>
      FInfo(
        resolution = 0.01,
        min = -3.38953e+38,
        max = 3.38953e+38,
        eps = 0.0078125,
        smallest_normal = 1.17549e-38,
        tiny = 1.17549e-38,
        dtype = "bfloat16"
      )

    case _: Float32 | _: Complex64 =>
      FInfo(
        resolution = 1e-06,
        min = -3.40282e+38,
        max = 3.40282e+38,
        eps = 1.19209e-07,
        smallest_normal = 1.17549e-38,
        tiny = 1.17549e-38,
        dtype = "float32"
      )
    case _: Float64 | _: Complex128 =>
      FInfo(
        resolution = 1e-15,
        min = -1.79769e+308,
        max = 1.79769e+308,
        eps = 2.22045e-16,
        smallest_normal = 2.22507e-308,
        tiny = 2.22507e-308,
        dtype = "float64"
      )

case class IInfo(
    resolution: Double,
    min: Double,
    max: Double,
    eps: Double,
    smallest_normal: Double,
    tiny: Double,
    dtype: String
)

// TODO iinfo
