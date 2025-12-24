#!/bin/bash

# Build script for Ubuntu 22.04 (Linux amd64)
# This script cross-compiles the Go program for Linux from macOS/other platforms

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

BINARY_NAME="write_fts"
SOURCE_FILE="write_full_text_search.go"
OUTPUT_DIR="build"
OUTPUT_FILE="$OUTPUT_DIR/${BINARY_NAME}_linux_amd64"

echo "=========================================================================="
echo -e "${BLUE}Building Turbopuffer FTS Write Tool for Ubuntu 22.04${NC}"
echo "=========================================================================="
echo ""

# Check if Go is installed
if ! command -v go &> /dev/null; then
    echo "❌ Error: Go is not installed"
    echo "Please install Go from: https://golang.org/dl/"
    exit 1
fi

echo -e "${GREEN}✅ Go version: $(go version)${NC}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if source file exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo "❌ Error: Source file '$SOURCE_FILE' not found"
    exit 1
fi

echo "📦 Building for Linux amd64 (Ubuntu 22.04)..."
echo "   Source: $SOURCE_FILE"
echo "   Output: $OUTPUT_FILE"
echo ""

# Build for Linux amd64
# GOOS=linux: target operating system (Linux)
# GOARCH=amd64: target architecture (64-bit x86)
# CGO_ENABLED=0: disable CGO for static binary (no external dependencies)
# -ldflags="-s -w": strip debug info and symbol table (smaller binary)
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build \
    -ldflags="-s -w" \
    -o "$OUTPUT_FILE" \
    "$SOURCE_FILE"

if [ $? -eq 0 ]; then
    # Get file size
    FILE_SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
    
    echo ""
    echo -e "${GREEN}✅ Build successful!${NC}"
    echo ""
    echo "=========================================================================="
    echo "Build Information:"
    echo "=========================================================================="
    echo "Binary:     $OUTPUT_FILE"
    echo "Size:       $FILE_SIZE"
    echo "Platform:   Linux amd64 (Ubuntu 22.04 compatible)"
    echo "OS:         $(uname -s)"
    echo "Arch:       $(uname -m)"
    echo ""
    echo "To deploy to Ubuntu 22.04:"
    echo "  1. Copy the binary: scp $OUTPUT_FILE user@ubuntu-server:/usr/local/bin/$BINARY_NAME"
    echo "  2. Make executable: ssh user@ubuntu-server 'chmod +x /usr/local/bin/$BINARY_NAME'"
    echo "  3. Run: ssh user@ubuntu-server '$BINARY_NAME -help'"
    echo ""
    echo "Or copy to current directory:"
    echo "  cp $OUTPUT_FILE ./${BINARY_NAME}_ubuntu"
    echo "=========================================================================="
else
    echo ""
    echo "❌ Build failed!"
    exit 1
fi
