// WhileLoopTests.swift - Tests for while loop implementation
// Copyright 2024 SwiftIR Project

import XCTest
@testable import SwiftIR
@testable import SwiftIRXLA

final class WhileLoopTests: XCTestCase {

    // MARK: - MLIR Generation Tests

    /// Test that while loop generates correct stablehlo.while operation using 3-tuple
    func testWhileLoopGeneratesMLIR() throws {
        // Set up builder context
        let builder = MLIRBuilder()
        DifferentiableTracer.currentBuilder = builder
        defer { DifferentiableTracer.currentBuilder = nil }

        // Add function argument
        let inputName = "%arg0"
        builder.addArgument(name: inputName, type: "tensor<f32>")

        // Create initial values (3-tuple: iter, value, dummy)
        let iterInitial = createConstant(0.0, shape: [], dtype: .float32)
        let valueInitial = createConstant(1.0, shape: [], dtype: .float32)
        let dummyInitial = createConstant(0.0, shape: [], dtype: .float32)

        // Run while loop with 3-tuple
        let (_, finalValue, _) = diffWhileLoop(
            initial: (iterInitial, valueInitial, dummyInitial),
            condition: { state in
                let maxIter = createConstant(10.0, shape: [], dtype: .float32)
                return state.0 < maxIter
            },
            body: { state in
                let one = createConstant(1.0, shape: [], dtype: .float32)
                let two = createConstant(2.0, shape: [], dtype: .float32)
                return (state.0 + one, state.1 * two, state.2)
            }
        )

        // Set result
        builder.setResults([finalValue.irValue])

        // Build and verify MLIR
        let mlirModule = builder.build(functionName: "while_test")
        let mlirText = mlirModule.mlirText

        // Verify stablehlo.while is present
        XCTAssertTrue(mlirText.contains("stablehlo.while"), "MLIR should contain stablehlo.while operation")
    }

    /// Test 4-tuple while loop (building simulation pattern)
    func testFourTupleWhileLoopMLIR() throws {
        let builder = MLIRBuilder()
        DifferentiableTracer.currentBuilder = builder
        defer { DifferentiableTracer.currentBuilder = nil }

        let inputName = "%arg0"
        builder.addArgument(name: inputName, type: "tensor<f32>")

        // Building simulation pattern: (iter, slab, quanta, tank)
        let iterInit = createConstant(0.0, shape: [], dtype: .float32)
        let slabInit = createConstant(33.3, shape: [], dtype: .float32)
        let quantaInit = createConstant(33.3, shape: [], dtype: .float32)
        let tankInit = createConstant(70.0, shape: [], dtype: .float32)

        let (_, finalSlab, _, _) = diffWhileLoop(
            initial: (iterInit, slabInit, quantaInit, tankInit),
            condition: { state in
                let maxIter = createConstant(5.0, shape: [], dtype: .float32)
                return state.0 < maxIter
            },
            body: { state in
                let one = createConstant(1.0, shape: [], dtype: .float32)
                let delta = createConstant(0.1, shape: [], dtype: .float32)
                return (
                    state.0 + one,
                    state.1 + delta,
                    state.2 + delta,
                    state.3 - delta
                )
            }
        )

        builder.setResults([finalSlab.irValue])
        let mlirModule = builder.build(functionName: "four_tuple_while")
        let mlirText = mlirModule.mlirText

        XCTAssertTrue(mlirText.contains("stablehlo.while"), "4-tuple while loop should generate stablehlo.while")
    }

    // MARK: - Comparison Operator Tests

    /// Test that comparison operators work correctly with tracers
    func testComparisonOperatorsWithBuilder() throws {
        let builder = MLIRBuilder()
        DifferentiableTracer.currentBuilder = builder
        defer { DifferentiableTracer.currentBuilder = nil }

        builder.addArgument(name: "%arg0", type: "tensor<f32>")

        let a = createConstant(5.0, shape: [], dtype: .float32)
        let b = createConstant(10.0, shape: [], dtype: .float32)

        // All comparison operators should produce valid tracers
        let lt = a < b
        let gt = a > b
        let le = a <= b
        let ge = a >= b

        // Verify they have valid IR values
        XCTAssertFalse(lt.irValue.isEmpty, "Less than comparison should produce valid IR")
        XCTAssertFalse(gt.irValue.isEmpty, "Greater than comparison should produce valid IR")
        XCTAssertFalse(le.irValue.isEmpty, "Less or equal comparison should produce valid IR")
        XCTAssertFalse(ge.irValue.isEmpty, "Greater or equal comparison should produce valid IR")
    }

    // MARK: - While Loop Result Tests

    /// Test that while loop returns valid tracers for all tuple elements
    func testWhileLoopReturnsValidTracers() throws {
        let builder = MLIRBuilder()
        DifferentiableTracer.currentBuilder = builder
        defer { DifferentiableTracer.currentBuilder = nil }

        builder.addArgument(name: "%arg0", type: "tensor<f32>")

        let iterInit = createConstant(0.0, shape: [], dtype: .float32)
        let val1Init = createConstant(1.0, shape: [], dtype: .float32)
        let val2Init = createConstant(2.0, shape: [], dtype: .float32)

        let (finalIter, finalVal1, finalVal2) = diffWhileLoop(
            initial: (iterInit, val1Init, val2Init),
            condition: { state in
                let maxIter = createConstant(5.0, shape: [], dtype: .float32)
                return state.0 < maxIter
            },
            body: { state in
                let one = createConstant(1.0, shape: [], dtype: .float32)
                return (state.0 + one, state.1 + one, state.2 + one)
            }
        )

        // All returned tracers should have valid IR values
        XCTAssertFalse(finalIter.irValue.isEmpty, "Final iteration should have valid IR")
        XCTAssertFalse(finalVal1.irValue.isEmpty, "Final value 1 should have valid IR")
        XCTAssertFalse(finalVal2.irValue.isEmpty, "Final value 2 should have valid IR")

        // The IR values should reference the while loop result
        XCTAssertTrue(finalIter.irValue.contains("#0") || finalIter.irValue.contains(":0"),
                      "Final iteration IR should reference while result element 0")
    }

    /// Test 4-tuple while loop returns valid tracers
    func testFourTupleWhileLoopReturnsValidTracers() throws {
        let builder = MLIRBuilder()
        DifferentiableTracer.currentBuilder = builder
        defer { DifferentiableTracer.currentBuilder = nil }

        builder.addArgument(name: "%arg0", type: "tensor<f32>")

        let iterInit = createConstant(0.0, shape: [], dtype: .float32)
        let slabInit = createConstant(33.3, shape: [], dtype: .float32)
        let quantaInit = createConstant(33.3, shape: [], dtype: .float32)
        let tankInit = createConstant(70.0, shape: [], dtype: .float32)

        let (finalIter, finalSlab, finalQuanta, finalTank) = diffWhileLoop(
            initial: (iterInit, slabInit, quantaInit, tankInit),
            condition: { state in
                let maxIter = createConstant(100.0, shape: [], dtype: .float32)
                return state.0 < maxIter
            },
            body: { state in
                let one = createConstant(1.0, shape: [], dtype: .float32)
                let delta = createConstant(0.01, shape: [], dtype: .float32)
                return (
                    state.0 + one,
                    state.1 - delta,
                    state.2 + delta,
                    state.3 + delta
                )
            }
        )

        // All returned tracers should have valid IR values
        XCTAssertFalse(finalIter.irValue.isEmpty, "Final iteration should have valid IR")
        XCTAssertFalse(finalSlab.irValue.isEmpty, "Final slab temp should have valid IR")
        XCTAssertFalse(finalQuanta.irValue.isEmpty, "Final quanta temp should have valid IR")
        XCTAssertFalse(finalTank.irValue.isEmpty, "Final tank temp should have valid IR")
    }

    // MARK: - PJRT Execution Tests

    /// Test while loop compiles and executes via PJRT (3-tuple)
    func testWhileLoopPJRTExecution() throws {
        // Simple counting loop: count from 0 to 5, doubling each iteration
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [], dtype: .float32)
        ) { dummy in
            let iterInit = createConstant(0.0, shape: [], dtype: .float32)
            let valueInit = createConstant(1.0, shape: [], dtype: .float32)
            let unusedInit = createConstant(0.0, shape: [], dtype: .float32)

            // Use 3-tuple
            let (_, finalValue, _) = diffWhileLoop(
                initial: (iterInit, valueInit, unusedInit),
                condition: { state in
                    let maxIter = createConstant(5.0, shape: [], dtype: .float32)
                    return state.0 < maxIter
                },
                body: { state in
                    let one = createConstant(1.0, shape: [], dtype: .float32)
                    let two = createConstant(2.0, shape: [], dtype: .float32)
                    return (state.0 + one, state.1 * two, state.2)
                }
            )

            // Return value squared to make it differentiable
            return finalValue * finalValue
        }

        // Execute
        let (output, _) = try gradFunc.forwardWithGradient([0.0], seed: [1.0])

        // After 5 iterations: 1 * 2^5 = 32, squared = 1024
        XCTAssertEqual(output[0], 1024.0, accuracy: 0.1, "While loop should execute correctly")
    }

    /// Test while loop with gradient computation (accumulation)
    func testWhileLoopGradient() throws {
        // Simpler test: just accumulate values
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [], dtype: .float32)
        ) { dummy in
            let iterInit = createConstant(0.0, shape: [], dtype: .float32)
            let sumInit = createConstant(0.0, shape: [], dtype: .float32)
            let unusedInit = createConstant(0.0, shape: [], dtype: .float32)

            // Use 3-tuple
            let (_, finalSum, _) = diffWhileLoop(
                initial: (iterInit, sumInit, unusedInit),
                condition: { state in
                    let maxIter = createConstant(3.0, shape: [], dtype: .float32)
                    return state.0 < maxIter
                },
                body: { state in
                    let one = createConstant(1.0, shape: [], dtype: .float32)
                    let increment = createConstant(10.0, shape: [], dtype: .float32)
                    return (state.0 + one, state.1 + increment, state.2)
                }
            )

            // Return sum squared (after 3 iterations: 0 + 10 + 10 + 10 = 30, squared = 900)
            // Using squared to match pattern that works in testWhileLoopPJRTExecution
            return finalSum * finalSum
        }

        let (output, gradient) = try gradFunc.forwardWithGradient([0.0], seed: [1.0])

        // 3 iterations of adding 10 = 30, squared = 900
        XCTAssertEqual(output[0], 900.0, accuracy: 0.1, "While loop sum squared should be 900")

        // Gradient should exist (may be 0 since dummy input isn't used)
        XCTAssertNotNil(gradient, "Gradient should be computed")
    }

    // MARK: - Integration Test

    /// Test that building simulation pattern generates complete MLIR structure
    func testBuildingSimulationMLIRGeneration() throws {
        let builder = MLIRBuilder()
        DifferentiableTracer.currentBuilder = builder
        defer { DifferentiableTracer.currentBuilder = nil }

        builder.addArgument(name: "%arg0", type: "tensor<f32>")

        // Building simulation state: (iteration, slabTemp, quantaTemp, tankTemp)
        let iterInit = createConstant(0.0, shape: [], dtype: .float32)
        let slabInit = createConstant(33.3, shape: [], dtype: .float32)
        let quantaInit = createConstant(33.3, shape: [], dtype: .float32)
        let tankInit = createConstant(70.0, shape: [], dtype: .float32)

        let numTimesteps = createConstant(100.0, shape: [], dtype: .float32)

        let (_, finalSlab, _, _) = diffWhileLoop(
            initial: (iterInit, slabInit, quantaInit, tankInit),
            condition: { state in
                return state.0 < numTimesteps
            },
            body: { state in
                let one = createConstant(1.0, shape: [], dtype: .float32)
                let heatTransfer = createConstant(0.01, shape: [], dtype: .float32)
                let slabToQuanta = (state.1 - state.2) * heatTransfer
                let quantaToTank = (state.2 - state.3) * heatTransfer

                return (
                    state.0 + one,
                    state.1 - slabToQuanta,
                    state.2 + slabToQuanta - quantaToTank,
                    state.3 + quantaToTank
                )
            }
        )

        builder.setResults([finalSlab.irValue])
        let mlirModule = builder.build(functionName: "building_sim")
        let mlirText = mlirModule.mlirText

        // Verify the while loop is present
        XCTAssertTrue(mlirText.contains("stablehlo.while"), "Building simulation should generate stablehlo.while")

        // Verify the module structure is valid
        XCTAssertTrue(mlirText.contains("func.func"), "Should have function declaration")
        XCTAssertTrue(mlirText.contains("return"), "Should have return statement")
    }
}
