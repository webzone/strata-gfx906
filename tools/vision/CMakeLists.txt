# tools/vision - strata-vision, the image encoder for Strata's multimodal path (optional; setup.py builds it when
# the vision encoder is chosen).  Uses llama.cpp's mtmd library at the same pinned commit as the engine's ggml.
#
#   cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=third_party/llama.cpp
#         [-DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native] [-DSTRATA_PORTABLE=OFF]
#         [or -DSTRATA_VISION_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx906]
#   cmake --build build-vision --target strata-vision
#
# Portable (AVX2 baseline) is the default, so an encoder built without extra flags runs on any AVX2 PC: the 0.1.32
# release's encoder was built native on an AVX-512 machine and stopped with an illegal instruction elsewhere
# (#411 #412 #419).  setup.py, which builds for the PC it runs on, passes -DSTRATA_PORTABLE=OFF.  The option is
# kept in CMakeCache.txt, so a build folder configured before this default still needs the flag (or a fresh folder).
cmake_minimum_required(VERSION 3.21)
project(strata_vision C CXX)

set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/llama.cpp" CACHE PATH "llama.cpp checkout")
option(STRATA_VISION_CUDA "run the vision encoder on the GPU (builds ggml-cuda)" OFF)
option(STRATA_VISION_HIP "run the experimental gfx906 vision encoder on HIP" OFF)
# Accepted for backward compatibility only. STRATA_EXPERIMENTAL_GFX906 is the switch of the deprecated
# gfx906 TEXT backend (main project, setup.py). The image encoder needs nothing but STRATA_VISION_HIP.
option(STRATA_EXPERIMENTAL_GFX906 "deprecated alias, unused by strata_vision; keep STRATA_VISION_HIP=ON" OFF)
if(STRATA_VISION_HIP)
  # HIP objects must be PIC when linked into Linux's default PIE executable.
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  if(STRATA_VISION_CUDA OR NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "HIP vision requires Linux and STRATA_VISION_CUDA=OFF")
  endif()
  if(NOT CMAKE_HIP_ARCHITECTURES STREQUAL "gfx906")
    message(FATAL_ERROR "HIP vision requires CMAKE_HIP_ARCHITECTURES=gfx906 (Vega20: MI50 / MI60)")
  endif()
  if(STRATA_EXPERIMENTAL_GFX906)
    message(STATUS "STRATA_EXPERIMENTAL_GFX906 is not needed by strata_vision; it selects the deprecated gfx906 text backend elsewhere")
  endif()
  # Vega has no MFMA/BF16 matrix instructions. Use GCN kernels and FP32 hipBLAS.
  set(GGML_HIP_NO_VMM ON CACHE BOOL "" FORCE)
  set(GGML_HIP_MMQ_MFMA OFF CACHE BOOL "" FORCE)
  set(GGML_HIP_GRAPHS OFF CACHE BOOL "" FORCE)
  set(GGML_CUDA_FA OFF CACHE BOOL "" FORCE)
endif()
option(STRATA_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" ON)

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
  # nothing above AVX2 either, even when a -D asks for it (a 9900K or a Zen 3 has neither AVX-512 nor AVX-VNNI)
  foreach(_f GGML_AVX512 GGML_AVX512_VBMI GGML_AVX512_VNNI GGML_AVX512_BF16 GGML_AVX_VNNI
             GGML_AMX_TILE GGML_AMX_INT8 GGML_AMX_BF16)
    set(${_f} OFF CACHE BOOL "" FORCE)
  endforeach()
  message(STATUS "strata-vision: portable build (AVX2 baseline, runs on other PCs)")
else()
  # this PC's instruction set; a build folder that was portable before gets a fresh native build's settings back
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  set(GGML_OPENMP ON CACHE BOOL "" FORCE)
  foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${_f} OFF CACHE BOOL "" FORCE)
  endforeach()
  message(STATUS "strata-vision: native build for this CPU (STRATA_PORTABLE=OFF) - not for other PCs")
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
