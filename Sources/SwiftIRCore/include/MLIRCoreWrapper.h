#ifndef MLIR_CORE_WRAPPER_H
#define MLIR_CORE_WRAPPER_H

#include <string.h>
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/Diagnostics.h>
#include <mlir-c/Dialect/Arith.h>
#include <mlir-c/Dialect/Func.h>
#include <mlir-c/Dialect/SCF.h>
#include <mlir-c/Dialect/Tensor.h>
#include <mlir-c/Dialect/Linalg.h>
#include <mlir-c/Pass.h>
#include <mlir-c/ExecutionEngine.h>
#include <mlir-c/Conversion.h>
#include <mlir/Dialect/Linalg/Passes.capi.h.inc>

#ifdef __cplusplus
extern "C" {
#endif

// Context wrappers
static inline MlirContext mlirContextCreateWrapper(void) {
    return mlirContextCreate();
}

static inline void mlirContextDestroyWrapper(MlirContext context) {
    mlirContextDestroy(context);
}

static inline bool mlirContextIsNullWrapper(MlirContext context) {
    return mlirContextIsNull(context);
}

// Module wrappers
static inline MlirModule mlirModuleCreateEmptyWrapper(MlirLocation location) {
    return mlirModuleCreateEmpty(location);
}

static inline void mlirModuleDestroyWrapper(MlirModule module) {
    mlirModuleDestroy(module);
}

static inline MlirOperation mlirModuleGetOperationWrapper(MlirModule module) {
    return mlirModuleGetOperation(module);
}

static inline bool mlirModuleIsNullWrapper(MlirModule module) {
    return mlirModuleIsNull(module);
}

// Location wrappers
static inline MlirLocation mlirLocationUnknownGetWrapper(MlirContext context) {
    return mlirLocationUnknownGet(context);
}

static inline MlirLocation mlirLocationFileLineColGetWrapper(
    MlirContext context,
    MlirStringRef filename,
    unsigned line,
    unsigned col
) {
    return mlirLocationFileLineColGet(context, filename, line, col);
}

// String utilities
static inline MlirStringRef mlirStringRefCreateWrapper(const char *str, size_t length) {
    MlirStringRef ref;
    ref.data = str;
    ref.length = length;
    return ref;
}

// Operation printing
static inline void mlirOperationPrintWrapper(
    MlirOperation op,
    MlirStringCallback callback,
    void *userData
) {
    mlirOperationPrint(op, callback, userData);
}

static inline bool mlirOperationVerifyWrapper(MlirOperation op) {
    return mlirOperationVerify(op);
}

// Type wrappers
static inline bool mlirTypeIsNullWrapper(MlirType type) {
    return mlirTypeIsNull(type);
}

static inline MlirType mlirIntegerTypeGetWrapper(MlirContext ctx, unsigned bitwidth) {
    return mlirIntegerTypeGet(ctx, bitwidth);
}

static inline MlirType mlirIntegerTypeSignedGetWrapper(MlirContext ctx, unsigned bitwidth) {
    return mlirIntegerTypeSignedGet(ctx, bitwidth);
}

static inline unsigned mlirIntegerTypeGetWidthWrapper(MlirType type) {
    return mlirIntegerTypeGetWidth(type);
}

static inline bool mlirIntegerTypeIsSignedWrapper(MlirType type) {
    return mlirIntegerTypeIsSigned(type);
}

static inline MlirType mlirF16TypeGetWrapper(MlirContext ctx) {
    return mlirF16TypeGet(ctx);
}

static inline MlirType mlirF32TypeGetWrapper(MlirContext ctx) {
    return mlirF32TypeGet(ctx);
}

static inline MlirType mlirF64TypeGetWrapper(MlirContext ctx) {
    return mlirF64TypeGet(ctx);
}

static inline bool mlirTypeIsAF16Wrapper(MlirType type) {
    return mlirTypeIsAF16(type);
}

static inline bool mlirTypeIsAF32Wrapper(MlirType type) {
    return mlirTypeIsAF32(type);
}

static inline bool mlirTypeIsAF64Wrapper(MlirType type) {
    return mlirTypeIsAF64(type);
}

static inline MlirType mlirIndexTypeGetWrapper(MlirContext ctx) {
    return mlirIndexTypeGet(ctx);
}

// Tensor type wrappers
static inline MlirType mlirRankedTensorTypeGetWrapper(
    MlirContext ctx,
    intptr_t rank,
    const int64_t *shape,
    MlirType elementType,
    MlirAttribute encoding
) {
    return mlirRankedTensorTypeGet(rank, shape, elementType, encoding);
}

static inline MlirType mlirUnrankedTensorTypeGetWrapper(
    MlirContext ctx,
    MlirType elementType
) {
    return mlirUnrankedTensorTypeGet(elementType);
}

static inline bool mlirTypeIsATensorWrapper(MlirType type) {
    return mlirTypeIsATensor(type);
}

static inline bool mlirTypeIsARankedTensorWrapper(MlirType type) {
    return mlirTypeIsARankedTensor(type);
}

static inline intptr_t mlirShapedTypeGetRankWrapper(MlirType type) {
    return mlirShapedTypeGetRank(type);
}

static inline bool mlirShapedTypeHasStaticShapeWrapper(MlirType type) {
    return mlirShapedTypeHasStaticShape(type);
}

static inline int64_t mlirShapedTypeGetDimSizeWrapper(MlirType type, intptr_t dim) {
    return mlirShapedTypeGetDimSize(type, dim);
}

static inline MlirType mlirShapedTypeGetElementTypeWrapper(MlirType type) {
    return mlirShapedTypeGetElementType(type);
}

// MemRef type wrappers
static inline MlirType mlirMemRefTypeGetWrapper(
    MlirContext ctx,
    MlirType elementType,
    intptr_t rank,
    const int64_t *shape,
    MlirAttribute layout,
    MlirAttribute memorySpace
) {
    return mlirMemRefTypeGet(elementType, rank, shape, layout, memorySpace);
}

static inline bool mlirTypeIsAMemRefWrapper(MlirType type) {
    return mlirTypeIsAMemRef(type);
}

// Vector type wrappers
static inline MlirType mlirVectorTypeGetWrapper(
    intptr_t rank,
    const int64_t *shape,
    MlirType elementType
) {
    return mlirVectorTypeGet(rank, shape, elementType);
}

static inline bool mlirTypeIsAVectorWrapper(MlirType type) {
    return mlirTypeIsAVector(type);
}

// Attribute wrappers (needed for encoding)
static inline bool mlirAttributeIsNullWrapper(MlirAttribute attr) {
    return mlirAttributeIsNull(attr);
}

static inline MlirAttribute mlirAttributeGetNullWrapper(void) {
    MlirAttribute attr;
    attr.ptr = NULL;
    return attr;
}

// Block and Region wrappers
static inline MlirRegion mlirRegionCreateWrapper(void) {
    return mlirRegionCreate();
}

static inline MlirBlock mlirBlockCreateWrapper(intptr_t nArgs, MlirType const *args, MlirLocation const *locs) {
    return mlirBlockCreate(nArgs, args, locs);
}

static inline void mlirRegionAppendOwnedBlockWrapper(MlirRegion region, MlirBlock block) {
    mlirRegionAppendOwnedBlock(region, block);
}

static inline void mlirBlockAppendOwnedOperationWrapper(MlirBlock block, MlirOperation operation) {
    mlirBlockAppendOwnedOperation(block, operation);
}

// Operation state and creation wrappers
static inline MlirOperationState mlirOperationStateGetWrapper(
    MlirStringRef name,
    MlirLocation location
) {
    return mlirOperationStateGet(name, location);
}

static inline void mlirOperationStateAddResultsWrapper(
    MlirOperationState *state,
    intptr_t n,
    MlirType const *results
) {
    mlirOperationStateAddResults(state, n, results);
}

static inline void mlirOperationStateAddOperandsWrapper(
    MlirOperationState *state,
    intptr_t n,
    MlirValue const *operands
) {
    mlirOperationStateAddOperands(state, n, operands);
}

static inline void mlirOperationStateAddOwnedRegionsWrapper(
    MlirOperationState *state,
    intptr_t n,
    MlirRegion const *regions
) {
    mlirOperationStateAddOwnedRegions(state, n, regions);
}

static inline void mlirOperationStateAddAttributesWrapper(
    MlirOperationState *state,
    intptr_t n,
    MlirNamedAttribute const *attributes
) {
    mlirOperationStateAddAttributes(state, n, attributes);
}

static inline MlirOperation mlirOperationCreateWrapper(MlirOperationState *state) {
    return mlirOperationCreate(state);
}

// Value wrappers
static inline MlirType mlirValueGetTypeWrapper(MlirValue value) {
    return mlirValueGetType(value);
}

static inline bool mlirValueIsNullWrapper(MlirValue value) {
    return mlirValueIsNull(value);
}

static inline MlirValue mlirOperationGetResultWrapper(MlirOperation op, intptr_t pos) {
    return mlirOperationGetResult(op, pos);
}

static inline MlirValue mlirBlockGetArgumentWrapper(MlirBlock block, intptr_t pos) {
    return mlirBlockGetArgument(block, pos);
}

static inline intptr_t mlirOperationGetNumResultsWrapper(MlirOperation op) {
    return mlirOperationGetNumResults(op);
}

static inline MlirBlock mlirOperationGetBlockWrapper(MlirOperation op) {
    return mlirOperationGetBlock(op);
}

// Dialect registration
static inline void mlirContextLoadAllAvailableDialectsWrapper(MlirContext context) {
    mlirContextLoadAllAvailableDialects(context);
}

static inline MlirDialect mlirContextGetOrLoadDialectWrapper(
    MlirContext context,
    MlirStringRef name
) {
    return mlirContextGetOrLoadDialect(context, name);
}

// Integer and Float attribute wrappers
static inline MlirAttribute mlirIntegerAttrGetWrapper(MlirType type, int64_t value) {
    return mlirIntegerAttrGet(type, value);
}

static inline MlirAttribute mlirFloatAttrDoubleGetWrapper(MlirContext ctx, MlirType type, double value) {
    return mlirFloatAttrDoubleGet(ctx, type, value);
}

static inline MlirAttribute mlirStringAttrGetWrapper(MlirContext ctx, MlirStringRef str) {
    return mlirStringAttrGet(ctx, str);
}

static inline MlirNamedAttribute mlirNamedAttributeGetWrapper(MlirIdentifier name, MlirAttribute attr) {
    return mlirNamedAttributeGet(name, attr);
}

static inline MlirIdentifier mlirIdentifierGetWrapper(MlirContext context, MlirStringRef str) {
    return mlirIdentifierGet(context, str);
}

static inline MlirStringRef mlirIdentifierStrWrapper(MlirIdentifier ident) {
    return mlirIdentifierStr(ident);
}

// Function type wrappers
static inline MlirType mlirFunctionTypeGetWrapper(
    MlirContext ctx,
    intptr_t numInputs,
    MlirType const *inputs,
    intptr_t numResults,
    MlirType const *results
) {
    return mlirFunctionTypeGet(ctx, numInputs, inputs, numResults, results);
}

static inline MlirAttribute mlirTypeAttrGetWrapper(MlirType type) {
    return mlirTypeAttrGet(type);
}

// Module body access
static inline MlirBlock mlirModuleGetBodyWrapper(MlirModule module) {
    MlirOperation moduleOp = mlirModuleGetOperation(module);
    MlirRegion bodyRegion = mlirOperationGetRegion(moduleOp, 0);
    return mlirRegionGetFirstBlock(bodyRegion);
}

// Dialect registration wrappers
static inline void mlirRegisterArithDialectWrapper(MlirContext context) {
    MlirDialectHandle handle = mlirGetDialectHandle__arith__();
    mlirDialectHandleRegisterDialect(handle, context);
    mlirDialectHandleLoadDialect(handle, context);
}

static inline void mlirRegisterFuncDialectWrapper(MlirContext context) {
    MlirDialectHandle handle = mlirGetDialectHandle__func__();
    mlirDialectHandleRegisterDialect(handle, context);
    mlirDialectHandleLoadDialect(handle, context);
}

static inline void mlirRegisterSCFDialectWrapper(MlirContext context) {
    MlirDialectHandle handle = mlirGetDialectHandle__scf__();
    mlirDialectHandleRegisterDialect(handle, context);
    mlirDialectHandleLoadDialect(handle, context);
}

static inline void mlirRegisterTensorDialectWrapper(MlirContext context) {
    MlirDialectHandle handle = mlirGetDialectHandle__tensor__();
    mlirDialectHandleRegisterDialect(handle, context);
    mlirDialectHandleLoadDialect(handle, context);
}

static inline void mlirRegisterLinalgDialectWrapper(MlirContext context) {
    MlirDialectHandle handle = mlirGetDialectHandle__linalg__();
    mlirDialectHandleRegisterDialect(handle, context);
    mlirDialectHandleLoadDialect(handle, context);
}

// Execution Engine wrappers
static inline bool mlirExecutionEngineIsNullWrapper(MlirExecutionEngine engine) {
    return mlirExecutionEngineIsNull(engine);
}

// String ref helper
static inline MlirStringRef mlirStringRefCreateFromCStringWrapper(const char *str) {
    MlirStringRef ref;
    ref.data = str;
    ref.length = strlen(str);
    return ref;
}

// Operation attribute setters
static inline void mlirOperationSetDiscardableAttributeByNameWrapper(
    MlirOperation op,
    MlirStringRef name,
    MlirAttribute attr
) {
    mlirOperationSetDiscardableAttributeByName(op, name, attr);
}

static inline MlirAttribute mlirOperationGetAttributeByNameWrapper(
    MlirOperation op,
    MlirStringRef name
) {
    return mlirOperationGetAttributeByName(op, name);
}

// Unit attribute (used for flags like llvm.emit_c_interface)
static inline MlirAttribute mlirUnitAttrGetWrapper(MlirContext ctx) {
    return mlirUnitAttrGet(ctx);
}

// Array attributes
static inline MlirAttribute mlirArrayAttrGetWrapper(MlirContext ctx, intptr_t numElements, MlirAttribute const *elements) {
    return mlirArrayAttrGet(ctx, numElements, elements);
}

static inline MlirAttribute mlirDenseI64ArrayAttrGetWrapper(MlirContext ctx, intptr_t numElements, int64_t const *values) {
    return mlirDenseI64ArrayGet(ctx, numElements, values);
}

// Operation iteration
static inline bool mlirOperationIsNullWrapper(MlirOperation op) {
    return mlirOperationIsNull(op);
}

static inline MlirOperation mlirBlockGetFirstOperationWrapper(MlirBlock block) {
    return mlirBlockGetFirstOperation(block);
}

static inline MlirOperation mlirOperationGetNextInBlockWrapper(MlirOperation op) {
    return mlirOperationGetNextInBlock(op);
}

static inline MlirRegion mlirOperationGetRegionWrapper(MlirOperation op, intptr_t pos) {
    return mlirOperationGetRegion(op, pos);
}

static inline MlirBlock mlirRegionGetFirstBlockWrapper(MlirRegion region) {
    return mlirRegionGetFirstBlock(region);
}

static inline intptr_t mlirOperationGetNumRegionsWrapper(MlirOperation op) {
    return mlirOperationGetNumRegions(op);
}

static inline MlirIdentifier mlirOperationGetNameWrapper(MlirOperation op) {
    return mlirOperationGetName(op);
}

// Logical result wrappers
static inline bool mlirLogicalResultIsSuccessWrapper(MlirLogicalResult result) {
    return mlirLogicalResultIsSuccess(result);
}

// StableHLO dialect registration (from unified library)
// These are implemented in cmake/SwiftIRMLIR.cpp
extern void swiftir_register_stablehlo_dialect(MlirContext context);
extern void swiftir_register_chlo_dialect(MlirContext context);
extern void swiftir_register_all_stablehlo_passes(void);

static inline void mlirRegisterStablehloDialectWrapper(MlirContext context) {
    swiftir_register_stablehlo_dialect(context);
}

static inline void mlirRegisterChloDialectWrapper(MlirContext context) {
    swiftir_register_chlo_dialect(context);
}

static inline void mlirRegisterAllStablehloPassesWrapper(void) {
    swiftir_register_all_stablehlo_passes();
}

// GPU dialect registration (needed for Linalg → GPU → SPIR-V lowering)
#include "mlir-c/Dialect/GPU.h"

static inline void mlirRegisterGPUDialectWrapper(MlirContext context) {
    MlirDialectHandle handle = mlirGetDialectHandle__gpu__();
    mlirDialectHandleRegisterDialect(handle, context);
}

// SPIR-V dialect registration
#include "mlir-c/Dialect/SPIRV.h"

static inline void mlirRegisterSPIRVDialectWrapper(MlirContext context) {
    MlirDialectHandle handle = mlirGetDialectHandle__spirv__();
    mlirDialectHandleRegisterDialect(handle, context);
}

// Register all Linalg passes (including convert-linalg-to-affine-loops, convert-linalg-to-parallel-loops)
static inline void mlirRegisterAllLinalgPassesWrapper(void) {
    mlirRegisterLinalgConvertLinalgToParallelLoopsPass();
    mlirRegisterLinalgConvertLinalgToAffineLoopsPass();
    // Register other Linalg passes as needed
}

// Register all GPU passes (including gpu-map-parallel-loops, convert-parallel-loops-to-gpu, gpu-kernel-outlining)
static inline void mlirRegisterAllGPUPassesWrapper(void) {
    mlirRegisterGPUPasses();
}

// Register all Conversion passes (including all conversion passes)
static inline void mlirRegisterAllConversionPassesWrapper(void) {
    mlirRegisterConversionPasses();
}

#ifdef __cplusplus
}
#endif

#endif // MLIR_CORE_WRAPPER_H
