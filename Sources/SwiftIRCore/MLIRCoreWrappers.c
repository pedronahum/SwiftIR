//===-- MLIRCoreWrappers.c - Non-inline C wrapper functions ----*- C -*-===//
//
// Implementation of wrapper functions that cannot be inline
//
//===------------------------------------------------------------------===//

#include "include/MLIRCoreWrapper.h"

// These functions are needed because Swift can't call inline functions via @_silgen_name

void mlirOperationSetDiscardableAttributeByNameNonInline(
    MlirOperation op,
    MlirStringRef name,
    MlirAttribute attr
) {
    mlirOperationSetDiscardableAttributeByName(op, name, attr);
}

MlirAttribute mlirOperationGetAttributeByNameNonInline(
    MlirOperation op,
    MlirStringRef name
) {
    return mlirOperationGetAttributeByName(op, name);
}

MlirAttribute mlirUnitAttrGetNonInline(MlirContext ctx) {
    return mlirUnitAttrGet(ctx);
}

bool mlirOperationIsNullNonInline(MlirOperation op) {
    return mlirOperationIsNull(op);
}

MlirOperation mlirBlockGetFirstOperationNonInline(MlirBlock block) {
    return mlirBlockGetFirstOperation(block);
}

MlirOperation mlirOperationGetNextInBlockNonInline(MlirOperation op) {
    return mlirOperationGetNextInBlock(op);
}

MlirRegion mlirOperationGetRegionNonInline(MlirOperation op, intptr_t pos) {
    return mlirOperationGetRegion(op, pos);
}

MlirBlock mlirRegionGetFirstBlockNonInline(MlirRegion region) {
    return mlirRegionGetFirstBlock(region);
}

intptr_t mlirOperationGetNumRegionsNonInline(MlirOperation op) {
    return mlirOperationGetNumRegions(op);
}

MlirIdentifier mlirOperationGetNameNonInline(MlirOperation op) {
    return mlirOperationGetName(op);
}
