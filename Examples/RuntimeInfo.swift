// RuntimeInfo.swift - Print SwiftIR runtime detection info
// Run with: swift run RuntimeInfo

import SwiftIRRuntime

print("=== SwiftIR Runtime Detection ===\n")

// Print runtime info
let info = RuntimeDetector.getRuntimeInfo()
print(info.summary)

print("\n=== Available Accelerators ===")
for acc in AcceleratorType.allCases {
    let available = PJRTClientFactory.isAvailable(acc)
    let marker = available ? "✓" : "✗"
    print("\(marker) \(acc.displayName): \(available)")
    if let path = RuntimeDetector.findPluginPath(for: acc) {
        print("  → \(path)")
    }
}

print("\n=== Detection Result ===")
let detected = RuntimeDetector.detect()
print("Best accelerator: \(detected)")
