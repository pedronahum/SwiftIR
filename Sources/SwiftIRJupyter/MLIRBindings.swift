// SwiftIRJupyter - MLIR Bindings via dlopen
// Pure Swift implementation - no C++ interop

import Foundation

// MARK: - MLIR Opaque Types (Swift wrappers)

/// Opaque pointer to MLIR context
public struct MLIRContextRef: Equatable {
    public let ptr: UnsafeMutableRawPointer?

    public init(_ ptr: UnsafeMutableRawPointer?) {
        self.ptr = ptr
    }

    public var isNull: Bool { ptr == nil }

    public static func == (lhs: MLIRContextRef, rhs: MLIRContextRef) -> Bool {
        lhs.ptr == rhs.ptr
    }
}

/// Opaque pointer to MLIR module
public struct MLIRModuleRef: Equatable {
    public let ptr: UnsafeMutableRawPointer?

    public init(_ ptr: UnsafeMutableRawPointer?) {
        self.ptr = ptr
    }

    public var isNull: Bool { ptr == nil }
}

/// Opaque pointer to MLIR location
public struct MLIRLocationRef {
    public let ptr: UnsafeMutableRawPointer?

    public init(_ ptr: UnsafeMutableRawPointer?) {
        self.ptr = ptr
    }

    public var isNull: Bool { ptr == nil }
}

/// Opaque pointer to MLIR operation
public struct MLIROperationRef {
    public let ptr: UnsafeMutableRawPointer?

    public init(_ ptr: UnsafeMutableRawPointer?) {
        self.ptr = ptr
    }

    public var isNull: Bool { ptr == nil }
}

/// Opaque pointer to MLIR type
public struct MLIRTypeRef {
    public let ptr: UnsafeMutableRawPointer?

    public init(_ ptr: UnsafeMutableRawPointer?) {
        self.ptr = ptr
    }

    public var isNull: Bool { ptr == nil }
}

/// Opaque pointer to MLIR value
public struct MLIRValueRef {
    public let ptr: UnsafeMutableRawPointer?

    public init(_ ptr: UnsafeMutableRawPointer?) {
        self.ptr = ptr
    }

    public var isNull: Bool { ptr == nil }
}

/// Opaque pointer to MLIR block
public struct MLIRBlockRef {
    public let ptr: UnsafeMutableRawPointer?

    public init(_ ptr: UnsafeMutableRawPointer?) {
        self.ptr = ptr
    }

    public var isNull: Bool { ptr == nil }
}

/// Opaque pointer to MLIR region
public struct MLIRRegionRef {
    public let ptr: UnsafeMutableRawPointer?

    public init(_ ptr: UnsafeMutableRawPointer?) {
        self.ptr = ptr
    }

    public var isNull: Bool { ptr == nil }
}

/// Opaque pointer to MLIR attribute
public struct MLIRAttributeRef {
    public let ptr: UnsafeMutableRawPointer?

    public init(_ ptr: UnsafeMutableRawPointer?) {
        self.ptr = ptr
    }

    public var isNull: Bool { ptr == nil }
}

// MARK: - MLIR Bindings

/// Dynamic bindings to MLIR C API
public final class MLIRBindings: @unchecked Sendable {

    /// Shared instance
    nonisolated(unsafe) public static let shared = MLIRBindings()

    /// Whether bindings are loaded
    public private(set) var isLoaded = false

    // Raw function pointers - we use UnsafeMutableRawPointer and cast at call site
    private var _contextCreate: UnsafeMutableRawPointer?
    private var _contextDestroy: UnsafeMutableRawPointer?
    private var _contextIsNull: UnsafeMutableRawPointer?
    private var _contextLoadAllDialects: UnsafeMutableRawPointer?
    private var _locationUnknownGet: UnsafeMutableRawPointer?
    private var _moduleCreateEmpty: UnsafeMutableRawPointer?
    private var _moduleDestroy: UnsafeMutableRawPointer?
    private var _moduleGetOperation: UnsafeMutableRawPointer?
    private var _moduleIsNull: UnsafeMutableRawPointer?
    private var _operationPrint: UnsafeMutableRawPointer?
    private var _operationVerify: UnsafeMutableRawPointer?
    private var _operationIsNull: UnsafeMutableRawPointer?
    private var _integerTypeGet: UnsafeMutableRawPointer?
    private var _f32TypeGet: UnsafeMutableRawPointer?
    private var _f64TypeGet: UnsafeMutableRawPointer?
    private var _indexTypeGet: UnsafeMutableRawPointer?
    private var _typeIsNull: UnsafeMutableRawPointer?

    private init() {}

    /// Load MLIR bindings from the native library
    public func load() throws {
        guard !isLoaded else { return }

        let loader = LibraryLoader.shared

        // Load JupyterMLIRWrapper which provides the wrapper functions
        try loader.load("JupyterMLIRWrapper")

        // Load function pointers using JupyterMLIR_ wrapper functions
        _contextCreate = try loader.symbol("JupyterMLIR_contextCreate")
        _contextDestroy = try loader.symbol("JupyterMLIR_contextDestroy")
        _contextIsNull = try loader.symbol("JupyterMLIR_contextIsNull")
        _contextLoadAllDialects = try loader.symbol("JupyterMLIR_contextLoadAllDialects")
        _locationUnknownGet = try loader.symbol("JupyterMLIR_locationUnknownGet")
        _moduleCreateEmpty = try loader.symbol("JupyterMLIR_moduleCreateEmpty")
        _moduleDestroy = try loader.symbol("JupyterMLIR_moduleDestroy")
        _moduleGetOperation = try loader.symbol("JupyterMLIR_moduleGetOperation")
        _moduleIsNull = try loader.symbol("JupyterMLIR_moduleIsNull")
        _operationPrint = try loader.symbol("JupyterMLIR_operationPrint")
        _operationVerify = try loader.symbol("JupyterMLIR_operationVerify")
        _operationIsNull = try loader.symbol("JupyterMLIR_operationIsNull")
        _integerTypeGet = try loader.symbol("JupyterMLIR_integerTypeGet")
        _f32TypeGet = try loader.symbol("JupyterMLIR_f32TypeGet")
        _f64TypeGet = try loader.symbol("JupyterMLIR_f64TypeGet")
        _indexTypeGet = try loader.symbol("JupyterMLIR_indexTypeGet")
        _typeIsNull = try loader.symbol("JupyterMLIR_typeIsNull")

        isLoaded = true
    }

    // MARK: - Context API

    public func contextCreate() throws -> MLIRContextRef {
        guard let fn = _contextCreate else {
            throw SwiftIRJupyterError.invalidState(message: "MLIR bindings not loaded")
        }
        typealias FnType = @convention(c) () -> UnsafeMutableRawPointer?
        let typedFn = unsafeBitCast(fn, to: FnType.self)
        return MLIRContextRef(typedFn())
    }

    public func contextDestroy(_ ctx: MLIRContextRef) {
        guard let fn = _contextDestroy else { return }
        typealias FnType = @convention(c) (UnsafeMutableRawPointer?) -> Void
        let typedFn = unsafeBitCast(fn, to: FnType.self)
        typedFn(ctx.ptr)
    }

    public func contextIsNull(_ ctx: MLIRContextRef) -> Bool {
        guard let fn = _contextIsNull else { return true }
        typealias FnType = @convention(c) (UnsafeMutableRawPointer?) -> Bool
        let typedFn = unsafeBitCast(fn, to: FnType.self)
        return typedFn(ctx.ptr)
    }

    public func contextLoadAllDialects(_ ctx: MLIRContextRef) {
        guard let fn = _contextLoadAllDialects else { return }
        typealias FnType = @convention(c) (UnsafeMutableRawPointer?) -> Void
        let typedFn = unsafeBitCast(fn, to: FnType.self)
        typedFn(ctx.ptr)
    }

    // MARK: - Location API

    public func locationUnknownGet(_ ctx: MLIRContextRef) throws -> MLIRLocationRef {
        guard let fn = _locationUnknownGet else {
            throw SwiftIRJupyterError.invalidState(message: "MLIR bindings not loaded")
        }
        typealias FnType = @convention(c) (UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer?
        let typedFn = unsafeBitCast(fn, to: FnType.self)
        return MLIRLocationRef(typedFn(ctx.ptr))
    }

    // MARK: - Module API

    public func moduleCreateEmpty(_ loc: MLIRLocationRef) throws -> MLIRModuleRef {
        guard let fn = _moduleCreateEmpty else {
            throw SwiftIRJupyterError.invalidState(message: "MLIR bindings not loaded")
        }
        typealias FnType = @convention(c) (UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer?
        let typedFn = unsafeBitCast(fn, to: FnType.self)
        return MLIRModuleRef(typedFn(loc.ptr))
    }

    public func moduleDestroy(_ module: MLIRModuleRef) {
        guard let fn = _moduleDestroy else { return }
        typealias FnType = @convention(c) (UnsafeMutableRawPointer?) -> Void
        let typedFn = unsafeBitCast(fn, to: FnType.self)
        typedFn(module.ptr)
    }

    public func moduleGetOperation(_ module: MLIRModuleRef) throws -> MLIROperationRef {
        guard let fn = _moduleGetOperation else {
            throw SwiftIRJupyterError.invalidState(message: "MLIR bindings not loaded")
        }
        typealias FnType = @convention(c) (UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer?
        let typedFn = unsafeBitCast(fn, to: FnType.self)
        return MLIROperationRef(typedFn(module.ptr))
    }

    public func moduleIsNull(_ module: MLIRModuleRef) -> Bool {
        guard let fn = _moduleIsNull else { return true }
        typealias FnType = @convention(c) (UnsafeMutableRawPointer?) -> Bool
        let typedFn = unsafeBitCast(fn, to: FnType.self)
        return typedFn(module.ptr)
    }

    // MARK: - Operation API

    public func operationPrint(_ op: MLIROperationRef) throws -> String {
        guard let fn = _operationPrint else {
            throw SwiftIRJupyterError.invalidState(message: "MLIR bindings not loaded")
        }

        // The callback receives the printed string
        var result = ""

        typealias CallbackType = @convention(c) (UnsafeRawPointer?, Int, UnsafeMutableRawPointer?) -> Void
        typealias FnType = @convention(c) (UnsafeMutableRawPointer?, CallbackType, UnsafeMutableRawPointer?) -> Void

        let callback: CallbackType = { data, length, userData in
            guard let data = data, let userData = userData else { return }
            let resultPtr = userData.assumingMemoryBound(to: String.self)
            let buffer = UnsafeBufferPointer(start: data.assumingMemoryBound(to: UInt8.self), count: length)
            resultPtr.pointee += String(decoding: buffer, as: UTF8.self)
        }

        let typedFn = unsafeBitCast(fn, to: FnType.self)
        withUnsafeMutablePointer(to: &result) { ptr in
            typedFn(op.ptr, callback, UnsafeMutableRawPointer(ptr))
        }

        return result
    }

    public func operationVerify(_ op: MLIROperationRef) -> Bool {
        guard let fn = _operationVerify else { return false }
        typealias FnType = @convention(c) (UnsafeMutableRawPointer?) -> Bool
        let typedFn = unsafeBitCast(fn, to: FnType.self)
        return typedFn(op.ptr)
    }

    public func operationIsNull(_ op: MLIROperationRef) -> Bool {
        guard let fn = _operationIsNull else { return true }
        typealias FnType = @convention(c) (UnsafeMutableRawPointer?) -> Bool
        let typedFn = unsafeBitCast(fn, to: FnType.self)
        return typedFn(op.ptr)
    }

    // MARK: - Type API

    public func integerTypeGet(_ ctx: MLIRContextRef, bitwidth: UInt32) throws -> MLIRTypeRef {
        guard let fn = _integerTypeGet else {
            throw SwiftIRJupyterError.invalidState(message: "MLIR bindings not loaded")
        }
        typealias FnType = @convention(c) (UnsafeMutableRawPointer?, UInt32) -> UnsafeMutableRawPointer?
        let typedFn = unsafeBitCast(fn, to: FnType.self)
        return MLIRTypeRef(typedFn(ctx.ptr, bitwidth))
    }

    public func f32TypeGet(_ ctx: MLIRContextRef) throws -> MLIRTypeRef {
        guard let fn = _f32TypeGet else {
            throw SwiftIRJupyterError.invalidState(message: "MLIR bindings not loaded")
        }
        typealias FnType = @convention(c) (UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer?
        let typedFn = unsafeBitCast(fn, to: FnType.self)
        return MLIRTypeRef(typedFn(ctx.ptr))
    }

    public func f64TypeGet(_ ctx: MLIRContextRef) throws -> MLIRTypeRef {
        guard let fn = _f64TypeGet else {
            throw SwiftIRJupyterError.invalidState(message: "MLIR bindings not loaded")
        }
        typealias FnType = @convention(c) (UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer?
        let typedFn = unsafeBitCast(fn, to: FnType.self)
        return MLIRTypeRef(typedFn(ctx.ptr))
    }

    public func indexTypeGet(_ ctx: MLIRContextRef) throws -> MLIRTypeRef {
        guard let fn = _indexTypeGet else {
            throw SwiftIRJupyterError.invalidState(message: "MLIR bindings not loaded")
        }
        typealias FnType = @convention(c) (UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer?
        let typedFn = unsafeBitCast(fn, to: FnType.self)
        return MLIRTypeRef(typedFn(ctx.ptr))
    }

    public func typeIsNull(_ type: MLIRTypeRef) -> Bool {
        guard let fn = _typeIsNull else { return true }
        typealias FnType = @convention(c) (UnsafeMutableRawPointer?) -> Bool
        let typedFn = unsafeBitCast(fn, to: FnType.self)
        return typedFn(type.ptr)
    }
}
