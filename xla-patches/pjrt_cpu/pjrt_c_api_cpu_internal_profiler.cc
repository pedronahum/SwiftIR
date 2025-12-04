/* Copyright 2023 The OpenXLA Authors.
   Modified by SwiftIR to add profiler extension support.

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

// SwiftIR: This is a modified version of pjrt_c_api_cpu_internal.cc
// that adds profiler extension support for capturing XLA internal metrics.
// It also exposes TraceMe API so Swift traces appear in the same profile
// as XLA internal traces.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/backends/profiler/plugin/plugin_tracer_impl.h"
#include "xla/backends/profiler/plugin/profiler_c_api.h"
#include "xla/backends/profiler/plugin/profiler_error.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_internal.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_layouts_extension.h"
#include "xla/pjrt/c/pjrt_c_api_memory_descriptions_extension.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_internal.h"
#include "xla/pjrt/c/pjrt_c_api_profiler_extension.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "tsl/profiler/lib/traceme.h"

namespace pjrt {
namespace cpu_plugin {

PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_Create_Args", PJRT_Client_Create_Args_STRUCT_SIZE,
      args->struct_size));

  xla::CpuClientOptions options;
  options.cpu_device_count = 4;

  if (args->create_options != nullptr) {
    absl::flat_hash_map<std::string, xla::PjRtValueType> create_options =
        ConvertFromPjRtNamedValueList(args->create_options, args->num_options);
    if (create_options.contains("cpu_device_count")) {
      int64_t device_count_option =
          std::get<int64_t>(create_options["cpu_device_count"]);
      options.cpu_device_count = device_count_option;
      LOG(INFO) << "cpu_device_count set via create_options: "
                << device_count_option;
    }
  }

  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                        xla::GetPjRtCpuClient(std::move(options)));
  args->client = pjrt::CreateWrapperClient(std::move(client));
  return nullptr;
}

PJRT_Error* PJRT_ExecuteContext_Create(PJRT_ExecuteContext_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_ExecuteContext_Create_Args",
      PJRT_ExecuteContext_Create_Args_STRUCT_SIZE, args->struct_size));
  auto execute_context = std::make_unique<xla::ExecuteContext>();
  args->context = pjrt::CreateWrapperExecuteContext(std::move(execute_context));
  return nullptr;
}

PJRT_Error* PJRT_CpuDeviceTopology_Create(
    PJRT_TopologyDescription_Create_Args* args) {
  return new PJRT_Error{
      absl::UnimplementedError("Topology not supported for CPU compilation.")};
}

// SwiftIR: Profiler API implementation
// This uses the same plugin_tracer_impl as the GPU plugin
PLUGIN_Profiler_Api profiler_api{
    /*struct_size=*/PLUGIN_Profiler_Api_STRUCT_SIZE,
    /*priv=*/nullptr,
    /*error_destroy=*/xla::profiler::PLUGIN_Profiler_Error_Destroy,
    /*error_message=*/xla::profiler::PLUGIN_Profiler_Error_Message,
    /*error_get_code=*/xla::profiler::PLUGIN_Profiler_Error_GetCode,
    /*create=*/xla::profiler::PLUGIN_Profiler_Create,
    /*destroy=*/xla::profiler::PLUGIN_Profiler_Destroy,
    /*start=*/xla::profiler::PLUGIN_Profiler_Start,
    /*stop=*/xla::profiler::PLUGIN_Profiler_Stop,
    /*collect_data=*/xla::profiler::PLUGIN_Profiler_CollectData,
};

// SwiftIR: Profiler extension registration
PJRT_Profiler_Extension profiler_extension{
    PJRT_Extension_Base{
        /*struct_size=*/PJRT_Profiler_Extension_STRUCT_SIZE,
        /*type=*/PJRT_Extension_Type::PJRT_Extension_Type_Profiler,
        /*next=*/nullptr,
    },
    /*profiler_api=*/&profiler_api,
};

const PJRT_Api* GetCpuPjrtApi() {
  static PJRT_Layouts_Extension layouts_extension =
      pjrt::CreateLayoutsExtension(nullptr);

  static PJRT_MemoryDescriptions_Extension memory_descriptions_extension =
      pjrt::CreateMemoryDescriptionsExtension(&layouts_extension.base);

  static PJRT_FFI_Extension ffi_extension =
      pjrt::CreateFfiExtension(&memory_descriptions_extension.base);

  static PJRT_PhaseCompile_Extension phase_compile_extension =
      pjrt::CreatePhaseCompileExtension(&ffi_extension.base,
                                        /*get_compiler=*/nullptr,
                                        /*destroy_compiler=*/nullptr);

  // SwiftIR: Link profiler extension into the extension chain
  profiler_extension.base.next = &phase_compile_extension.base;

  static const PJRT_Api pjrt_api = pjrt::CreatePjrtApi(
      pjrt::cpu_plugin::PJRT_Client_Create,
      pjrt::cpu_plugin::PJRT_ExecuteContext_Create,
      pjrt::cpu_plugin::PJRT_CpuDeviceTopology_Create,
      pjrt::PJRT_Plugin_Initialize_NoOp, &profiler_extension.base,
      pjrt::PJRT_Plugin_Attributes_Xla);

  return &pjrt_api;
}

}  // namespace cpu_plugin
}  // namespace pjrt

// SwiftIR: Export the GetPjrtApi function and TraceMe API
extern "C" {

const PJRT_Api* GetPjrtApi() {
  return pjrt::cpu_plugin::GetCpuPjrtApi();
}

// ============================================================================
// SwiftIR TraceMe C API
// These functions allow Swift to add trace events to the SAME TraceMeRecorder
// that XLA uses internally, so all traces appear in a unified profile.
// ============================================================================

// Start a TraceMe activity. Returns activity_id (or 0 if tracing disabled).
// The activity_id is used to pair start/end events in the trace.
// level: 1 = critical (always shown), 2 = info, 3 = verbose
int64_t PJRT_TraceMeStart(const char* name, int32_t level) {
  return tsl::profiler::TraceMe::ActivityStart(
      absl::string_view(name), static_cast<int>(level));
}

// End a TraceMe activity started by PJRT_TraceMeStart.
void PJRT_TraceMeStop(int64_t activity_id) {
  tsl::profiler::TraceMe::ActivityEnd(activity_id);
}

// Check if tracing is active at the given level.
// Useful to avoid expensive string operations when tracing is disabled.
bool PJRT_TraceMeActive(int32_t level) {
  return tsl::profiler::TraceMe::Active(static_cast<int>(level));
}

// Record an instant event (no duration, just a point in time).
void PJRT_TraceMeInstant(const char* name, int32_t level) {
  if (tsl::profiler::TraceMe::Active(static_cast<int>(level))) {
    tsl::profiler::TraceMe::InstantActivity(
        [name]() { return std::string(name); },
        static_cast<int>(level));
  }
}

}  // extern "C"
