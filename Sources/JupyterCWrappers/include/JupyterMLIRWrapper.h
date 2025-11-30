#ifndef JUPYTER_MLIR_WRAPPER_H
#define JUPYTER_MLIR_WRAPPER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// These wrappers use raw pointers instead of MLIR struct types
// so Swift can use them via dlopen without needing the MLIR header layout

// Context functions - return/accept void* instead of MlirContext
void* JupyterMLIR_contextCreate(void);
void JupyterMLIR_contextDestroy(void* ctx);
bool JupyterMLIR_contextIsNull(void* ctx);
void JupyterMLIR_contextLoadAllDialects(void* ctx);

// Location functions
void* JupyterMLIR_locationUnknownGet(void* ctx);

// Module functions
void* JupyterMLIR_moduleCreateEmpty(void* loc);
void JupyterMLIR_moduleDestroy(void* module);
void* JupyterMLIR_moduleGetOperation(void* module);
bool JupyterMLIR_moduleIsNull(void* module);

// Operation functions
typedef void (*JupyterMLIR_PrintCallback)(const char* data, size_t length, void* userData);
void JupyterMLIR_operationPrint(void* op, JupyterMLIR_PrintCallback callback, void* userData);
bool JupyterMLIR_operationVerify(void* op);
bool JupyterMLIR_operationIsNull(void* op);

// Type functions
void* JupyterMLIR_integerTypeGet(void* ctx, unsigned bitwidth);
void* JupyterMLIR_f32TypeGet(void* ctx);
void* JupyterMLIR_f64TypeGet(void* ctx);
void* JupyterMLIR_indexTypeGet(void* ctx);
bool JupyterMLIR_typeIsNull(void* type);

#ifdef __cplusplus
}
#endif

#endif // JUPYTER_MLIR_WRAPPER_H
