#!/bin/bash

# Build script for search_horizon_perf Go version

set -e

echo "🏗️  Building search_horizon_perf (Go version)..."

# Check if Go is installed
if ! command -v go &> /dev/null; then
    echo "❌ Go is not installed. Please install Go 1.21 or later."
    exit 1
fi

# Check Go version
GO_VERSION=$(go version | cut -d' ' -f3 | cut -d'o' -f2)
echo "✅ Go version: $GO_VERSION"

# Initialize/update Go modules
echo "📦 Downloading dependencies..."
go mod download

# Build with optimizations
echo "🔨 Compiling binary..."
go build -ldflags="-s -w" -o search_horizon_perf search_horizon_perf.go

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "📁 Binary created: ./search_horizon_perf"
    echo ""
    echo "🚀 Usage examples:"
    echo "   ./search_horizon_perf                    # Default run"
    echo "   ./search_horizon_perf -workers=50        # 50 concurrent workers"
    echo "   ./search_horizon_perf -search-type=hybrid # Hybrid search"
    echo "   ./search_horizon_perf -help              # Show all options"
    echo ""
    
    # Show binary info
    if [ -f "./search_horizon_perf" ]; then
        BINARY_SIZE=$(du -h ./search_horizon_perf | cut -f1)
        echo "📊 Binary size: $BINARY_SIZE"
    fi
else
    echo "❌ Build failed!"
    exit 1
fi
