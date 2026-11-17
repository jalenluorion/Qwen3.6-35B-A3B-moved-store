// swift-tools-version: 6.2
// phone-store G4 native runtime. Build with xcodebuild (SwiftPM alone cannot compile MLX's Metal shaders):
//   cd native && ./build.sh            (see docs/G4-runtime.md)
import PackageDescription

let package = Package(
    name: "phone-store-native",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "PhoneStore", targets: ["PhoneStore"]),
        .executable(name: "phone-store-run", targets: ["phone-store-run"]),
    ],
    dependencies: [
        // mlx-swift 0.32.3 vendored with mlx core v0.32.3 (the Python runner's mlx), see vendor/VENDOR.md
        .package(path: "vendor/mlx-swift"),
        // tokenizer + chat template (Jinja)
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.4.0"),
    ],
    targets: [
        .target(
            name: "PhoneStore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "Cmlx", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            swiftSettings: [.unsafeFlags(["-Ounchecked"], .when(configuration: .release))]
        ),
        .executableTarget(
            name: "phone-store-run",
            dependencies: ["PhoneStore"]
        ),
    ],
    swiftLanguageModes: [.v5]
)
