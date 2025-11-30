// JupyterMLIRWrapper.c - MLIR wrapper functions for Jupyter/REPL usage
// Uses raw pointers so Swift can call via dlopen without needing struct layouts

#include <mlir-c/IR.h>
#include <mlir-c/Support.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/RegisterEverything.h>
#include <string.h>

// Context functions

void* JupyterMLIR_contextCreate(void) {
    MlirContext ctx = mlirContextCreate();
    // Return the internal pointer directly (cast away const)
    return (void*)ctx.ptr;
}

void JupyterMLIR_contextDestroy(void* ctx) {
    MlirContext mlirCtx;
    mlirCtx.ptr = ctx;
    mlirContextDestroy(mlirCtx);
}

bool JupyterMLIR_contextIsNull(void* ctx) {
    MlirContext mlirCtx;
    mlirCtx.ptr = ctx;
    return mlirContextIsNull(mlirCtx);
}

void JupyterMLIR_contextLoadAllDialects(void* ctx) {
    MlirContext mlirCtx;
    mlirCtx.ptr = ctx;

    // Load all available dialects that are already registered
    mlirContextLoadAllAvailableDialects(mlirCtx);
}

// Location functions

void* JupyterMLIR_locationUnknownGet(void* ctx) {
    MlirContext mlirCtx;
    mlirCtx.ptr = ctx;
    MlirLocation loc = mlirLocationUnknownGet(mlirCtx);
    return (void*)loc.ptr;
}

// Module functions

void* JupyterMLIR_moduleCreateEmpty(void* loc) {
    MlirLocation mlirLoc;
    mlirLoc.ptr = loc;
    MlirModule module = mlirModuleCreateEmpty(mlirLoc);
    return (void*)module.ptr;
}

void JupyterMLIR_moduleDestroy(void* module) {
    MlirModule mlirModule;
    mlirModule.ptr = module;
    mlirModuleDestroy(mlirModule);
}

void* JupyterMLIR_moduleGetOperation(void* module) {
    MlirModule mlirModule;
    mlirModule.ptr = module;
    MlirOperation op = mlirModuleGetOperation(mlirModule);
    return (void*)op.ptr;
}

bool JupyterMLIR_moduleIsNull(void* module) {
    MlirModule mlirModule;
    mlirModule.ptr = module;
    return mlirModuleIsNull(mlirModule);
}

// Operation functions

typedef void (*JupyterMLIR_PrintCallback)(const char* data, size_t length, void* userData);

// Adapter for MLIR's callback signature
static void printCallbackAdapter(MlirStringRef str, void* userData) {
    void** args = (void**)userData;
    JupyterMLIR_PrintCallback callback = (JupyterMLIR_PrintCallback)args[0];
    void* userDataInner = args[1];
    callback(str.data, str.length, userDataInner);
}

void JupyterMLIR_operationPrint(void* op, JupyterMLIR_PrintCallback callback, void* userData) {
    MlirOperation mlirOp;
    mlirOp.ptr = op;

    // Pack callback and userData into array
    void* args[2] = { (void*)callback, userData };

    mlirOperationPrint(mlirOp, printCallbackAdapter, args);
}

bool JupyterMLIR_operationVerify(void* op) {
    MlirOperation mlirOp;
    mlirOp.ptr = op;
    return mlirOperationVerify(mlirOp);
}

bool JupyterMLIR_operationIsNull(void* op) {
    MlirOperation mlirOp;
    mlirOp.ptr = op;
    return mlirOperationIsNull(mlirOp);
}

// Type functions

void* JupyterMLIR_integerTypeGet(void* ctx, unsigned bitwidth) {
    MlirContext mlirCtx;
    mlirCtx.ptr = ctx;
    MlirType type = mlirIntegerTypeGet(mlirCtx, bitwidth);
    return (void*)type.ptr;
}

void* JupyterMLIR_f32TypeGet(void* ctx) {
    MlirContext mlirCtx;
    mlirCtx.ptr = ctx;
    MlirType type = mlirF32TypeGet(mlirCtx);
    return (void*)type.ptr;
}

void* JupyterMLIR_f64TypeGet(void* ctx) {
    MlirContext mlirCtx;
    mlirCtx.ptr = ctx;
    MlirType type = mlirF64TypeGet(mlirCtx);
    return (void*)type.ptr;
}

void* JupyterMLIR_indexTypeGet(void* ctx) {
    MlirContext mlirCtx;
    mlirCtx.ptr = ctx;
    MlirType type = mlirIndexTypeGet(mlirCtx);
    return (void*)type.ptr;
}

bool JupyterMLIR_typeIsNull(void* type) {
    MlirType mlirType;
    mlirType.ptr = type;
    return mlirTypeIsNull(mlirType);
}
