/// Shardy C API Wrapper for Swift
/// This header provides access to the Shardy (SDY) C API for tensor sharding.

#ifndef SDY_CAPI_WRAPPER_H
#define SDY_CAPI_WRAPPER_H

#include <stdint.h>
#include <stdbool.h>
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// SDY Dialect Registration
// =============================================================================

/// Register all SDY passes and pipelines with MLIR.
void mlirRegisterAllSdyPassesAndPipelines(void);

/// Get the SDY dialect handle for registration.
MlirDialectHandle mlirGetDialectHandle__sdy__(void);

// =============================================================================
// MeshAxisAttr - Represents a single axis of a device mesh
// =============================================================================

/// Check if an attribute is a MeshAxisAttr.
bool sdyAttributeIsAMeshAxisAttr(MlirAttribute attr);

/// Create a MeshAxisAttr with the given name and size.
MlirAttribute sdyMeshAxisAttrGet(MlirContext ctx, MlirStringRef name, int64_t size);

/// Get the name of a MeshAxisAttr.
MlirStringRef sdyMeshAxisAttrGetName(MlirAttribute attr);

/// Get the size of a MeshAxisAttr.
int64_t sdyMeshAxisAttrGetSize(MlirAttribute attr);

// =============================================================================
// MeshAttr - Represents a device mesh topology
// =============================================================================

/// Check if an attribute is a MeshAttr.
bool sdyAttributeIsAMeshAttr(MlirAttribute attr);

/// Create a MeshAttr with axes and optional device IDs.
MlirAttribute sdyMeshAttrGet(MlirContext ctx, intptr_t nAxes,
                             const MlirAttribute *axes, intptr_t nDeviceIds,
                             const int64_t *deviceIds);

/// Get the number of device IDs in a MeshAttr.
int64_t sdyMeshAttrGetDeviceIdsSize(MlirAttribute attr);

/// Get a device ID at a specific position.
int64_t sdyMeshAttrGetDeviceIdsElem(MlirAttribute attr, int64_t pos);

/// Get the number of axes in a MeshAttr.
intptr_t sdyMeshAttrGetAxesSize(MlirAttribute attr);

/// Get an axis at a specific position.
MlirAttribute sdyMeshAttrGetAxesElem(MlirAttribute attr, intptr_t pos);

// =============================================================================
// SubAxisInfoAttr - Sub-axis information for hierarchical partitioning
// =============================================================================

/// Check if an attribute is a SubAxisInfoAttr.
bool sdyAttributeIsASubAxisInfoAttr(MlirAttribute attr);

/// Create a SubAxisInfoAttr with preSize and size.
MlirAttribute sdySubAxisInfoAttrGet(MlirContext ctx, int64_t preSize,
                                    int64_t size);

/// Get the preSize of a SubAxisInfoAttr.
int64_t sdySubAxisInfoAttrGetPreSize(MlirAttribute attr);

/// Get the size of a SubAxisInfoAttr.
int64_t sdySubAxisInfoAttrGetSize(MlirAttribute attr);

// =============================================================================
// AxisRefAttr - Reference to a mesh axis
// =============================================================================

/// Check if an attribute is an AxisRefAttr.
bool sdyAttributeIsAnAxisRefAttr(MlirAttribute attr);

/// Create an AxisRefAttr. Pass null subAxisInfo if not needed.
MlirAttribute sdyAxisRefAttrGet(MlirContext ctx, MlirStringRef name,
                                MlirAttribute subAxisInfo);

/// Get the name of an AxisRefAttr.
MlirStringRef sdyAxisRefAttrGetName(MlirAttribute attr);

/// Get the SubAxisInfo of an AxisRefAttr (null if none).
MlirAttribute sdyAxisRefAttrGetSubAxisInfo(MlirAttribute attr);

// =============================================================================
// DimensionShardingAttr - Sharding specification for one dimension
// =============================================================================

/// Check if an attribute is a DimensionShardingAttr.
bool sdyAttributeIsADimensionShardingAttr(MlirAttribute attr);

/// Create a DimensionShardingAttr. Use -1 for priority if none.
MlirAttribute sdyDimensionShardingAttrGet(MlirContext ctx, intptr_t nAxes,
                                          const MlirAttribute *axes,
                                          bool isClosed, int64_t priority);

/// Get the number of axes in a DimensionShardingAttr.
intptr_t sdyDimensionShardingAttrGetAxesSize(MlirAttribute attr);

/// Get an axis at a specific position.
MlirAttribute sdyDimensionShardingAttrGetAxesElem(MlirAttribute attr,
                                                   intptr_t pos);

/// Check if the dimension is closed.
bool sdyDimensionShardingAttrGetIsClosed(MlirAttribute attr);

/// Get the priority (-1 if none).
int64_t sdyDimensionShardingAttrGetPriority(MlirAttribute attr);

// =============================================================================
// TensorShardingAttr - Complete sharding specification for a tensor
// =============================================================================

/// Check if an attribute is a TensorShardingAttr.
bool sdyAttributeIsATensorShardingAttr(MlirAttribute attr);

/// Create a TensorShardingAttr.
MlirAttribute sdyTensorShardingAttrGet(MlirContext ctx, MlirAttribute meshOrRef,
                                       intptr_t nDimShardings,
                                       const MlirAttribute *dimShardings,
                                       intptr_t nReplicatedAxes,
                                       const MlirAttribute *replicatedAxes,
                                       intptr_t nUnreducedAxes,
                                       const MlirAttribute *unreducedAxes);

/// Get the mesh or mesh reference.
MlirAttribute sdyTensorShardingAttrGetMeshOrRef(MlirAttribute attr);

/// Get the number of dimension shardings.
intptr_t sdyTensorShardingAttrGetDimShardingsSize(MlirAttribute attr);

/// Get a dimension sharding at a specific position.
MlirAttribute sdyTensorShardingAttrGetDimShardingsElem(MlirAttribute attr,
                                                        intptr_t pos);

/// Get the number of replicated axes.
intptr_t sdyTensorShardingAttrGetReplicatedAxesSize(MlirAttribute attr);

/// Get a replicated axis at a specific position.
MlirAttribute sdyTensorShardingAttrGetReplicatedAxesElem(MlirAttribute attr,
                                                          intptr_t pos);

/// Get the number of unreduced axes.
intptr_t sdyTensorShardingAttrGetUnreducedAxesSize(MlirAttribute attr);

/// Get an unreduced axis at a specific position.
MlirAttribute sdyTensorShardingAttrGetUnreducedAxesElem(MlirAttribute attr,
                                                         intptr_t pos);

// =============================================================================
// TensorShardingPerValueAttr - Shardings for multiple values
// =============================================================================

/// Check if an attribute is a TensorShardingPerValueAttr.
bool sdyAttributeIsATensorShardingPerValueAttr(MlirAttribute attr);

/// Create a TensorShardingPerValueAttr.
MlirAttribute sdyTensorShardingPerValueAttrGet(MlirContext ctx,
                                               intptr_t nShardings,
                                               const MlirAttribute *shardings);

/// Get the number of shardings.
intptr_t sdyTensorShardingPerValueAttrGetShardingsSize(MlirAttribute attr);

/// Get a sharding at a specific position.
MlirAttribute sdyTensorShardingPerValueAttrGetShardingsElem(MlirAttribute attr,
                                                             intptr_t pos);

// =============================================================================
// ManualAxesAttr - Manually partitioned axes
// =============================================================================

/// Check if an attribute is a ManualAxesAttr.
bool sdyAttributeIsAManualAxesAttr(MlirAttribute attr);

/// Create a ManualAxesAttr.
MlirAttribute sdyManualAxesAttrGet(MlirContext ctx, intptr_t nAxes,
                                   const MlirAttribute *axes);

/// Get the number of manual axes.
intptr_t sdyManualAxesAttrGetAxesSize(MlirAttribute attr);

/// Get an axis name at a specific position.
MlirStringRef sdyManualAxesAttrGetAxesElem(MlirAttribute attr, intptr_t pos);

// =============================================================================
// Helper Functions for Swift Integration
// =============================================================================

/// Create a null MlirAttribute (for optional parameters).
static inline MlirAttribute sdyNullAttribute(void) {
    MlirAttribute attr;
    attr.ptr = 0;
    return attr;
}

/// Check if an attribute is null.
static inline bool sdyAttributeIsNull(MlirAttribute attr) {
    return attr.ptr == 0;
}

/// Create an MlirStringRef from a C string.
static inline MlirStringRef sdyStringRefCreate(const char *str, size_t length) {
    MlirStringRef ref;
    ref.data = str;
    ref.length = length;
    return ref;
}

#ifdef __cplusplus
}
#endif

#endif // SDY_CAPI_WRAPPER_H
