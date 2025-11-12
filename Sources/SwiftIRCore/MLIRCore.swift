/// Core MLIR types and operations for Swift
import MLIRCoreWrapper

/// Errors that can occur during MLIR operations
public enum MLIRError: Error {
    case parsingFailed(String)
    case verificationFailed(String)
    case invalidOperation(String)

    public var localizedDescription: String {
        switch self {
        case .parsingFailed(let msg):
            return "MLIR parsing failed: \(msg)"
        case .verificationFailed(let msg):
            return "MLIR verification failed: \(msg)"
        case .invalidOperation(let msg):
            return "Invalid MLIR operation: \(msg)"
        }
    }
}

/// A wrapper around MlirContext that manages the lifetime of an MLIR context.
///
/// The MLIR context is the top-level object that holds all the IR state.
/// It owns all the memory for the IR and provides uniquing for types and attributes.
public final class MLIRContext {
    private var context: MlirContext

    /// Creates a new MLIR context.
    public init() {
        self.context = mlirContextCreateWrapper()
    }

    deinit {
        mlirContextDestroyWrapper(context)
    }

    /// Returns true if the context is null (invalid).
    public var isNull: Bool {
        mlirContextIsNullWrapper(context)
    }

    /// Gets the underlying MlirContext handle.
    /// - Warning: Use with caution. The context is owned by this Swift object.
    public var handle: MlirContext {
        context
    }
}

/// A wrapper around MlirModule representing an MLIR module.
///
/// A module is the top-level container for MLIR operations. It contains a single
/// region with a single block.
public final class MLIRModule {
    private var module: MlirModule
    public let context: MLIRContext

    /// Creates an empty MLIR module in the given context.
    /// - Parameter context: The MLIR context to create the module in.
    public init(context: MLIRContext = MLIRContext()) {
        self.context = context
        // Create an unknown location for the module
        let location = mlirLocationUnknownGetWrapper(context.handle)
        self.module = mlirModuleCreateEmptyWrapper(location)
    }

    /// Internal initializer for wrapping an existing MlirModule (for parsing)
    private init(wrapping module: MlirModule, context: MLIRContext) {
        self.module = module
        self.context = context
    }

    /// Parses an MLIR module from a string.
    /// - Parameters:
    ///   - mlirString: The MLIR textual representation to parse
    ///   - context: The MLIR context to parse the module in
    /// - Returns: A parsed MLIR module
    /// - Throws: MLIRError if parsing fails
    public static func parse(_ mlirString: String, context: MLIRContext) throws -> MLIRModule {
        let stringRef = mlirString.withCString { ptr in
            mlirStringRefCreateFromCString(ptr)
        }

        let parsedModule = mlirModuleCreateParse(context.handle, stringRef)

        // Check if parsing succeeded
        if mlirModuleIsNullWrapper(parsedModule) {
            throw MLIRError.parsingFailed("Failed to parse MLIR module")
        }

        // Create an MLIRModule wrapper
        return MLIRModule(wrapping: parsedModule, context: context)
    }

    deinit {
        if !mlirModuleIsNullWrapper(module) {
            mlirModuleDestroyWrapper(module)
        }
    }

    /// Returns true if the module is null (invalid).
    public var isNull: Bool {
        mlirModuleIsNullWrapper(module)
    }

    /// Gets the operation that this module wraps.
    public var operation: MLIROperation {
        let op = mlirModuleGetOperationWrapper(module)
        return MLIROperation(handle: op)
    }

    /// Gets the underlying MlirModule handle.
    /// - Warning: Use with caution. The module is owned by this Swift object.
    public var handle: MlirModule {
        module
    }

    /// Prints the module to a string.
    /// - Returns: The MLIR textual representation of the module.
    public func dump() -> String {
        return operation.dump()
    }

    /// Verifies the module.
    /// - Returns: true if the module is valid, false otherwise.
    public func verify() -> Bool {
        return operation.verify()
    }

    /// Gets the module's body block
    public func getBody() -> MLIRBlock {
        let blockHandle = mlirModuleGetBodyWrapper(module)
        return MLIRBlock(handle: blockHandle)
    }

    /// Appends an operation to the module body
    public func append(_ operation: MLIROperation) {
        let body = getBody()
        body.append(operation)
    }
}

/// A wrapper around MlirOperation representing an MLIR operation.
///
/// Operations are the fundamental unit of IR in MLIR. They have a name, attributes,
/// operands, results, and regions.
public struct MLIROperation {
    public let handle: MlirOperation

    /// Creates an operation wrapper from a handle.
    /// - Parameter handle: The underlying MlirOperation handle.
    public init(handle: MlirOperation) {
        self.handle = handle
    }

    /// Prints the operation to a string.
    /// - Returns: The MLIR textual representation of the operation.
    public func dump() -> String {
        var result = ""
        mlirOperationPrintWrapper(handle, { stringRef, userData in
            let data = stringRef.data!
            let length = Int(stringRef.length)
            // Reinterpret as UInt8 and create String
            let uint8Ptr = UnsafeRawPointer(data).assumingMemoryBound(to: UInt8.self)
            let buffer = UnsafeBufferPointer(start: uint8Ptr, count: length)
            if let string = String(decoding: buffer, as: UTF8.self) as String? {
                let resultPtr = userData!.assumingMemoryBound(to: String.self)
                resultPtr.pointee += string
            }
        }, &result)
        return result
    }

    /// Verifies the operation.
    /// - Returns: true if the operation is valid, false otherwise.
    public func verify() -> Bool {
        return mlirOperationVerifyWrapper(handle)
    }
}

/// A wrapper around MlirLocation representing a source location in MLIR.
///
/// Locations are used for debugging and error reporting. They can represent
/// file locations, unknown locations, or fused locations.
public struct MLIRLocation {
    internal let handle: MlirLocation

    /// Creates an unknown location in the given context.
    /// - Parameter context: The MLIR context.
    /// - Returns: An unknown location.
    public static func unknown(in context: MLIRContext) -> MLIRLocation {
        let location = mlirLocationUnknownGetWrapper(context.handle)
        return MLIRLocation(handle: location)
    }

    /// Creates a file location in the given context.
    /// - Parameters:
    ///   - context: The MLIR context.
    ///   - filename: The filename.
    ///   - line: The line number.
    ///   - column: The column number.
    /// - Returns: A file location.
    public static func file(
        _ filename: String,
        line: UInt,
        column: UInt,
        in context: MLIRContext
    ) -> MLIRLocation {
        let filenameRef = filename.withCString { ptr in
            mlirStringRefCreateWrapper(ptr, filename.utf8.count)
        }
        let location = mlirLocationFileLineColGetWrapper(
            context.handle,
            filenameRef,
            UInt32(line),
            UInt32(column)
        )
        return MLIRLocation(handle: location)
    }

    internal init(handle: MlirLocation) {
        self.handle = handle
    }
}

/// A wrapper around MlirValue representing an SSA value in MLIR.
///
/// Values are the results of operations or block arguments. They are typed
/// and can be used as operands to other operations.
public struct MLIRValue {
    internal let handle: MlirValue

    internal init(handle: MlirValue) {
        self.handle = handle
    }

    /// Gets the type of this value
    public func getType() -> MlirType {
        mlirValueGetTypeWrapper(handle)
    }

    /// Returns true if the value is null (invalid)
    public var isNull: Bool {
        mlirValueIsNullWrapper(handle)
    }
}

/// A wrapper around MlirBlock representing a basic block in MLIR.
///
/// Blocks contain a sequence of operations and can have arguments (block parameters).
public final class MLIRBlock {
    internal let handle: MlirBlock

    internal init(handle: MlirBlock) {
        self.handle = handle
    }

    /// Creates a new block with the specified arguments
    public init(arguments: [MlirType] = [], locations: [MLIRLocation] = [], context: MLIRContext) {
        var argTypes = arguments
        var argLocs = locations.map { $0.handle }

        // If no locations provided, use unknown locations
        if argLocs.isEmpty && !argTypes.isEmpty {
            argLocs = Array(repeating: MLIRLocation.unknown(in: context).handle, count: argTypes.count)
        }

        let argCount = argTypes.count
        self.handle = argTypes.withUnsafeMutableBufferPointer { argsPtr in
            argLocs.withUnsafeMutableBufferPointer { locsPtr in
                mlirBlockCreateWrapper(argCount, argsPtr.baseAddress, locsPtr.baseAddress)
            }
        }
    }

    /// Appends an operation to this block
    public func append(_ operation: MLIROperation) {
        mlirBlockAppendOwnedOperationWrapper(handle, operation.handle)
    }

    /// Gets a block argument at the specified position
    public func getArgument(_ index: Int) -> MLIRValue {
        MLIRValue(handle: mlirBlockGetArgumentWrapper(handle, index))
    }
}

/// A wrapper around MlirRegion representing a region in MLIR.
///
/// Regions contain a sequence of blocks and represent control flow graphs.
public final class MLIRRegion {
    internal let handle: MlirRegion

    /// Creates a new empty region
    public init() {
        self.handle = mlirRegionCreateWrapper()
    }

    internal init(handle: MlirRegion) {
        self.handle = handle
    }

    /// Appends a block to this region
    public func append(_ block: MLIRBlock) {
        mlirRegionAppendOwnedBlockWrapper(handle, block.handle)
    }
}

/// Builder for creating MLIR operations
public final class OperationBuilder {
    private var state: MlirOperationState
    private let context: MLIRContext
    private let operationName: String  // Keep string alive!
    private var attributeNames: [String] = []  // Keep attribute name strings alive!
    private var storedAttributes: [MlirNamedAttribute] = []  // Keep named attributes alive!

    /// Creates a new operation builder
    public init(name: String, location: MLIRLocation, context: MLIRContext) {
        self.context = context
        self.operationName = name
        // Create identifier first (this interns the string in MLIR's string pool)
        let identifier = name.withCString { ptr in
            let nameRef = mlirStringRefCreateWrapper(ptr, name.utf8.count)
            return mlirIdentifierGetWrapper(context.handle, nameRef)
        }
        // Get the string ref from the identifier (now safe to use, as it points to interned string)
        let nameRef = mlirIdentifierStrWrapper(identifier)
        self.state = mlirOperationStateGetWrapper(nameRef, location.handle)
    }

    /// Adds result types to the operation
    public func addResults(_ types: [MlirType]) -> OperationBuilder {
        var typesArray = types
        typesArray.withUnsafeMutableBufferPointer { ptr in
            mlirOperationStateAddResultsWrapper(&state, types.count, ptr.baseAddress)
        }
        return self
    }

    /// Adds operands to the operation
    public func addOperands(_ values: [MLIRValue]) -> OperationBuilder {
        var handles = values.map { $0.handle }
        handles.withUnsafeMutableBufferPointer { ptr in
            mlirOperationStateAddOperandsWrapper(&state, values.count, ptr.baseAddress)
        }
        return self
    }

    /// Adds regions to the operation
    public func addRegions(_ regions: [MLIRRegion]) -> OperationBuilder {
        var handles = regions.map { $0.handle }
        handles.withUnsafeMutableBufferPointer { ptr in
            mlirOperationStateAddOwnedRegionsWrapper(&state, regions.count, ptr.baseAddress)
        }
        return self
    }

    /// Adds attributes to the operation
    public func addAttributes(_ attributes: [(String, MlirAttribute)]) -> OperationBuilder {
        // Store attribute names to keep them alive
        let names = attributes.map { $0.0 }
        self.attributeNames.append(contentsOf: names)

        // Store the starting index for the new attributes
        let startIndex = storedAttributes.count

        // Create named attributes and add them to stored array
        for (name, attr) in attributes {
            let identifier = name.withCString { ptr in
                let nameRef = mlirStringRefCreateWrapper(ptr, name.utf8.count)
                return mlirIdentifierGetWrapper(context.handle, nameRef)
            }
            let namedAttr = mlirNamedAttributeGetWrapper(identifier, attr)
            storedAttributes.append(namedAttr)
        }

        // NOTE: We can't add attributes to the operation state here because
        // MlirOperationState just stores a pointer to the attributes array.
        // We must add them in build() while the storedAttributes array is still alive.

        return self
    }

    /// Builds and returns the operation
    public func build() -> MLIROperation {
        let attrCount = storedAttributes.count

        // CRITICAL: mlirOperationCreate must be called WHILE the attributes pointer is valid.
        // The MlirOperationState just stores a pointer to the attributes array, it doesn't copy it!
        // So we must call mlirOperationCreate inside the withUnsafeMutableBufferPointer closure.
        if attrCount > 0 {
            return storedAttributes.withUnsafeMutableBufferPointer { attrPtr in
                var localState = state
                mlirOperationStateAddAttributesWrapper(&localState, attrCount, attrPtr.baseAddress)
                return MLIROperation(handle: mlirOperationCreateWrapper(&localState))
            }
        } else {
            var localState = state
            return MLIROperation(handle: mlirOperationCreateWrapper(&localState))
        }
    }
}

// Extension to MLIROperation to get results
extension MLIROperation {
    /// Gets the number of results this operation produces
    public var numResults: Int {
        Int(mlirOperationGetNumResultsWrapper(handle))
    }

    /// Gets a result value at the specified index
    public func getResult(_ index: Int) -> MLIRValue {
        MLIRValue(handle: mlirOperationGetResultWrapper(handle, index))
    }

    /// Sets a discardable attribute on the operation
    /// Discardable attributes don't affect semantics and can be dropped during transformations
    public func setDiscardableAttribute(name: String, value: MlirAttribute) {
        let nameRef = name.withCString { ptr in
            mlirStringRefCreateWrapper(ptr, name.utf8.count)
        }
        mlirOperationSetDiscardableAttributeByNameWrapper(handle, nameRef, value)
    }

    /// Gets an attribute by name
    public func getAttribute(name: String) -> MlirAttribute {
        let nameRef = name.withCString { ptr in
            mlirStringRefCreateWrapper(ptr, name.utf8.count)
        }
        return mlirOperationGetAttributeByNameWrapper(handle, nameRef)
    }

    /// Gets the name of the operation as a string
    public var name: String {
        let identifier = mlirOperationGetNameWrapper(handle)
        let stringRef = mlirIdentifierStrWrapper(identifier)
        let data = stringRef.data!
        let length = Int(stringRef.length)
        let uint8Ptr = UnsafeRawPointer(data).assumingMemoryBound(to: UInt8.self)
        let buffer = UnsafeBufferPointer(start: uint8Ptr, count: length)
        return String(decoding: buffer, as: UTF8.self)
    }

    /// Checks if this operation is null
    public var isNull: Bool {
        mlirOperationIsNullWrapper(handle)
    }

    /// Gets the number of regions in this operation
    public var numRegions: Int {
        Int(mlirOperationGetNumRegionsWrapper(handle))
    }

    /// Gets a region at the specified index
    public func getRegion(_ index: Int) -> MLIRRegion {
        MLIRRegion(handle: mlirOperationGetRegionWrapper(handle, index))
    }
}

// Extension to MLIRBlock for iteration
extension MLIRBlock {
    /// Gets the first operation in the block
    public var firstOperation: MLIROperation? {
        let op = mlirBlockGetFirstOperationWrapper(handle)
        return mlirOperationIsNullWrapper(op) ? nil : MLIROperation(handle: op)
    }

    /// Iterates over all operations in the block
    public func forEach(_ body: (MLIROperation) -> Void) {
        var current = firstOperation
        while let op = current {
            body(op)
            let nextHandle = mlirOperationGetNextInBlockWrapper(op.handle)
            current = mlirOperationIsNullWrapper(nextHandle) ? nil : MLIROperation(handle: nextHandle)
        }
    }
}

// Extension to MLIRRegion for iteration
extension MLIRRegion {
    /// Gets the first block in the region
    public var firstBlock: MLIRBlock? {
        let blockHandle = mlirRegionGetFirstBlockWrapper(handle)
        // Check if block is null by checking if ptr is nil
        return blockHandle.ptr == nil ? nil : MLIRBlock(handle: blockHandle)
    }
}

// Extension to MLIRContext for dialect loading
extension MLIRContext {
    /// Loads all available dialects into this context
    public func loadAllDialects() {
        mlirContextLoadAllAvailableDialectsWrapper(handle)
    }

    /// Loads a specific dialect by name
    @discardableResult
    public func loadDialect(_ name: String) -> Bool {
        let nameRef = name.withCString { ptr in
            mlirStringRefCreateWrapper(ptr, name.utf8.count)
        }
        let dialect = mlirContextGetOrLoadDialectWrapper(handle, nameRef)
        return dialect.ptr != nil
    }

    /// Creates a unit attribute (used for boolean flags)
    /// Unit attributes represent the presence of an attribute without any value
    public func createUnitAttribute() -> MlirAttribute {
        return mlirUnitAttrGetWrapper(handle)
    }

    /// Registers and loads the Tensor dialect
    public func registerTensorDialect() {
        mlirRegisterTensorDialectWrapper(handle)
    }

    /// Registers and loads the Linalg dialect
    public func registerLinalgDialect() {
        mlirRegisterLinalgDialectWrapper(handle)
    }

    /// Registers and loads the StableHLO dialect
    public func registerStablehloDialect() {
        mlirRegisterStablehloDialectWrapper(handle)
    }

    /// Registers and loads the CHLO dialect
    public func registerChloDialect() {
        mlirRegisterChloDialectWrapper(handle)
    }

    /// Registers and loads the Func dialect
    /// Required for parsing func.func operations
    public func registerFuncDialect() {
        mlirRegisterFuncDialectWrapper(handle)
    }

    /// Registers and loads the GPU dialect
    /// Required for Linalg → GPU → SPIR-V lowering pipeline
    public func registerGPUDialect() {
        mlirRegisterGPUDialectWrapper(handle)
    }

    /// Registers and loads the SPIR-V dialect
    /// Target dialect for GPU execution via Vulkan/Metal
    public func registerSPIRVDialect() {
        mlirRegisterSPIRVDialectWrapper(handle)
    }

    /// Registers all Linalg passes
    /// Required for Linalg transformations like convert-linalg-to-parallel-loops
    public func registerAllLinalgPasses() {
        mlirRegisterAllLinalgPassesWrapper()
    }

    /// Registers all GPU passes
    /// Required for GPU transformations like gpu-map-parallel-loops, convert-parallel-loops-to-gpu
    public func registerAllGPUPasses() {
        mlirRegisterAllGPUPassesWrapper()
    }

    /// Registers all Conversion passes
    /// Required for all dialect conversion passes (Linalg→Standard, GPU→SPIR-V, etc.)
    public func registerAllConversionPasses() {
        mlirRegisterAllConversionPassesWrapper()
    }
}

// MARK: - C API Declarations

@_silgen_name("mlirOperationSetDiscardableAttributeByNameNonInline")
func mlirOperationSetDiscardableAttributeByNameWrapper(
    _ op: MlirOperation,
    _ name: MlirStringRef,
    _ attr: MlirAttribute
)

@_silgen_name("mlirOperationGetAttributeByNameNonInline")
func mlirOperationGetAttributeByNameWrapper(
    _ op: MlirOperation,
    _ name: MlirStringRef
) -> MlirAttribute

@_silgen_name("mlirUnitAttrGetNonInline")
func mlirUnitAttrGetWrapper(_ ctx: MlirContext) -> MlirAttribute

@_silgen_name("mlirOperationIsNullNonInline")
func mlirOperationIsNullWrapper(_ op: MlirOperation) -> Bool

@_silgen_name("mlirBlockGetFirstOperationNonInline")
func mlirBlockGetFirstOperationWrapper(_ block: MlirBlock) -> MlirOperation

@_silgen_name("mlirOperationGetNextInBlockNonInline")
func mlirOperationGetNextInBlockWrapper(_ op: MlirOperation) -> MlirOperation

@_silgen_name("mlirOperationGetRegionNonInline")
func mlirOperationGetRegionWrapper(_ op: MlirOperation, _ pos: Int) -> MlirRegion

@_silgen_name("mlirRegionGetFirstBlockNonInline")
func mlirRegionGetFirstBlockWrapper(_ region: MlirRegion) -> MlirBlock

@_silgen_name("mlirOperationGetNumRegionsNonInline")
func mlirOperationGetNumRegionsWrapper(_ op: MlirOperation) -> Int

@_silgen_name("mlirOperationGetNameNonInline")
func mlirOperationGetNameWrapper(_ op: MlirOperation) -> MlirIdentifier
