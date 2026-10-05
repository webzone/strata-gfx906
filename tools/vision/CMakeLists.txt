# tools/vision - strata-vision, the image encoder for Strata's multimodal path (optional; setup.py builds it when
# the vision encoder is chosen).  Uses llama.cpp's mtmd library at the same pinned commit as the engine's ggml.
#
#   cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=third_party/llama.cpp
#         [-DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native]
#         [or -DSTRATA_VISION_HIP=ON -DSTRATA_EXPERIMENTAL_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906]
#   cmake --build build-vision --target strata-vision
cmake_minimum_required(VERSION 3.21)
project(strata_vision C CXX)

set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/llama.cpp" CACHE PATH "llama.cpp checkout")
option(STRATA_VISION_CUDA "run the vision encoder on the GPU (builds ggml-cuda)" OFF)
option(STRATA_VISION_HIP "run the experimental gfx906 vision encoder on HIP" OFF)
option(STRATA_EXPERIMENTAL_GFX906 "explicitly enable experimental MI50/MI60 support" OFF)
if(STRATA_VISION_HIP)
  # HIP objects must be PIC when linked into Linux's default PIE executable.
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  if(STRATA_VISION_CUDA OR NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "HIP vision requires Linux and STRATA_VISION_CUDA=OFF")
  endif()
  if(NOT STRATA_EXPERIMENTAL_GFX906 OR NOT CMAKE_HIP_ARCHITECTURES STREQUAL "gfx906")
    message(FATAL_ERROR "HIP vision requires STRATA_EXPERIMENTAL_GFX906=ON and CMAKE_HIP_ARCHITECTURES=gfx906")
  endif()
  # Vega has no MFMA/BF16 matrix instructions. Use GCN kernels and FP32 hipBLAS.
  set(GGML_HIP_NO_VMM ON CACHE BOOL "" FORCE)
  set(GGML_HIP_MMQ_MFMA OFF CACHE BOOL "" FORCE)
  set(GGML_HIP_GRAPHS OFF CACHE BOOL "" FORCE)
  set(GGML_CUDA_FA OFF CACHE BOOL "" FORCE)
endif()
option(STRATA_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" OFF)
if(STRATA_PORTABLE AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
if(STRATA_PORTABLE)
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)        # no vcomp140.dll / libgomp needed on the user's PC
  foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${_f} ON CACHE BOOL "" FORCE)
  endforeach()
else()
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
endif()
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ${STRATA_VISION_CUDA} CACHE BOOL "" FORCE)
set(GGML_HIP ${STRATA_VISION_HIP} CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_DIR} llama EXCLUDE_FROM_ALL)

if(STRATA_VISION_HIP)
  # Vision-only overlays keep the pinned checkout byte-identical.
  set(_vision_cuda_dir "${LLAMA_DIR}/ggml/src/ggml-cuda")
  get_target_property(_vision_hip_sources ggml-hip SOURCES)
  # The pinned CPU GELU rounds its input/output through an FP16 lookup table.
  file(READ "${_vision_cuda_dir}/unary.cu" _vision_unary_source)
  set(_vision_gelu_anchor "    return ggml_cuda_op_gelu_single(x);")
  string(FIND "${_vision_unary_source}" "${_vision_gelu_anchor}" _vision_anchor_pos)
  if(_vision_anchor_pos EQUAL -1)
    message(FATAL_ERROR "Pinned ggml vision GELU anchor missing")
  endif()
  string(REPLACE "${_vision_gelu_anchor}"
    "    if (x <= -10.0f) return 0.0f;\n    if (x >= 10.0f) return x;\n    x = __half2float(__float2half_rn(x));\n    return __half2float(__float2half_rn(ggml_cuda_op_gelu_single(x)));"
    _vision_unary_source "${_vision_unary_source}")
  set(_vision_unary_overlay "${CMAKE_CURRENT_BINARY_DIR}/gfx906-vision-unary.cu")
  file(CONFIGURE OUTPUT "${_vision_unary_overlay}" CONTENT "${_vision_unary_source}" @ONLY)
  list(FILTER _vision_hip_sources EXCLUDE REGEX "/unary[.]cu$")
  list(APPEND _vision_hip_sources "${_vision_unary_overlay}")
  set_property(TARGET ggml-hip PROPERTY SOURCES "${_vision_hip_sources}")
  set_source_files_properties("${_vision_unary_overlay}" TARGET_DIRECTORY ggml-hip PROPERTIES LANGUAGE HIP
    INCLUDE_DIRECTORIES "${_vision_cuda_dir}")

  file(READ "${LLAMA_DIR}/tools/mtmd/clip.cpp" _vision_clip_source)
  foreach(_anchor "ggml_dup_tensor(ctx_clip.ctx_data.get(), cur)"
                  "if (ggml_backend_buft_is_host(buft)) {"
                  "total_data_size += ggml_nbytes(t);")
    string(FIND "${_vision_clip_source}" "${_anchor}" _vision_anchor_pos)
    if(_vision_anchor_pos EQUAL -1)
      message(FATAL_ERROR "Pinned mtmd vision weight-loading anchor missing: ${_anchor}")
    endif()
  endforeach()
  string(REPLACE "ggml_dup_tensor(ctx_clip.ctx_data.get(), cur)"
    "strata_vision_tensor(ctx_clip.ctx_data.get(), cur)" _vision_clip_source "${_vision_clip_source}")
  string(REPLACE "if (ggml_backend_buft_is_host(buft)) {"
    "if (t->type == GGML_TYPE_BF16) {\n                        strata_vision_read_bf16(fin, cur);\n                    } else if (ggml_backend_buft_is_host(buft)) {"
    _vision_clip_source "${_vision_clip_source}")
  string(REPLACE "total_data_size += ggml_nbytes(t);"
    "total_data_size += ggml_nbytes(ggml_get_tensor(ctx_clip.ctx_data.get(), t->name));"
    _vision_clip_source "${_vision_clip_source}")
  set(_vision_clip_overlay "${CMAKE_CURRENT_BINARY_DIR}/gfx906-vision-clip.cpp")
  file(CONFIGURE OUTPUT "${_vision_clip_overlay}"
    CONTENT "#include \"${CMAKE_CURRENT_SOURCE_DIR}/gfx906-vision.h\"\n${_vision_clip_source}" @ONLY)
  get_target_property(_vision_mtmd_sources mtmd SOURCES)
  list(FILTER _vision_mtmd_sources EXCLUDE REGEX "(^|/)clip[.]cpp$")
  list(APPEND _vision_mtmd_sources "${_vision_clip_overlay}")
  set_property(TARGET mtmd PROPERTY SOURCES "${_vision_mtmd_sources}")
endif()

add_executable(strata-vision strata_vision.cpp)
target_link_libraries(strata-vision PRIVATE mtmd llama)
target_compile_features(strata-vision PRIVATE cxx_std_17)
if(STRATA_VISION_HIP)
  find_package(hip REQUIRED)
  target_compile_definitions(strata-vision PRIVATE STRATA_VISION_HIP)
  target_link_libraries(strata-vision PRIVATE hip::host)
endif()
set_target_properties(strata-vision PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
