#!/bin/bash

# Quick setup script for running write_full_text_search.go on Ubuntu 22.04

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================================================="
echo -e "${BLUE}Turbopuffer FTS - Ubuntu 22.04 Quick Setup${NC}"
echo "=========================================================================="
echo ""

# Step 1: Check/Install Go
echo "Step 1: Checking Go installation..."
if ! command -v go &> /dev/null; then
    echo -e "${YELLOW}Go not found. Installing...${NC}"
    
    # Try apt first
    if command -v apt &> /dev/null; then
        sudo apt update
        sudo apt install -y golang-go
    # Try snap
    elif command -v snap &> /dev/null; then
        sudo snap install go --classic
    else
        echo -e "${RED}❌ Cannot install Go automatically. Please install manually.${NC}"
        echo "Visit: https://golang.org/dl/"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Go installed${NC}"
else
    GO_VERSION=$(go version | awk '{print $3}')
    echo -e "${GREEN}✅ Go already installed: $GO_VERSION${NC}"
fi

# Check Go version (need 1.21+)
GO_MAJOR=$(go version | awk '{print $3}' | sed 's/go//' | cut -d. -f1)
GO_MINOR=$(go version | awk '{print $3}' | sed 's/go//' | cut -d. -f2)

if [ "$GO_MAJOR" -lt 1 ] || ([ "$GO_MAJOR" -eq 1 ] && [ "$GO_MINOR" -lt 21 ]); then
    echo -e "${RED}❌ Go version too old. Need 1.21+. Current: $(go version)${NC}"
    echo "Please install a newer version from: https://golang.org/dl/"
    exit 1
fi

echo ""

# Step 2: Check source file
echo "Step 2: Checking source file..."
if [ ! -f "write_full_text_search.go" ]; then
    echo -e "${RED}❌ write_full_text_search.go not found in current directory${NC}"
    echo "Please run this script from the directory containing write_full_text_search.go"
    exit 1
fi
echo -e "${GREEN}✅ Source file found${NC}"
echo ""

# Step 3: Initialize Go module
echo "Step 3: Setting up Go module..."
if [ ! -f "go.mod" ]; then
    echo "Initializing Go module..."
    go mod init turbopuffer-fts 2>/dev/null || true
    echo -e "${GREEN}✅ Go module initialized${NC}"
else
    echo -e "${GREEN}✅ Go module already exists${NC}"
fi
echo ""

# Step 4: Install dependencies
echo "Step 4: Installing dependencies..."
echo "This may take a few minutes..."

go get github.com/segmentio/parquet-go 2>&1 | grep -v "go: downloading" || true
go get github.com/turbopuffer/turbopuffer-go 2>&1 | grep -v "go: downloading" || true
go mod tidy

echo -e "${GREEN}✅ Dependencies installed${NC}"
echo ""

# Step 5: Check API key
echo "Step 5: Checking API key..."
if [ -z "$TURBOPUFFER_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  TURBOPUFFER_API_KEY not set${NC}"
    echo ""
    read -p "Enter your Turbopuffer API key (or press Enter to skip): " API_KEY
    if [ -n "$API_KEY" ]; then
        export TURBOPUFFER_API_KEY="$API_KEY"
        echo -e "${GREEN}✅ API key set${NC}"
    else
        echo -e "${YELLOW}⚠️  API key not set. You'll need to set it before running.${NC}"
        echo "   export TURBOPUFFER_API_KEY=your_key"
    fi
else
    echo -e "${GREEN}✅ API key found${NC}"
fi
echo ""

# Step 6: Verify setup
echo "Step 6: Verifying setup..."
go vet write_full_text_search.go 2>&1 | head -5 || true
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""

# Summary
echo "=========================================================================="
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "=========================================================================="
echo ""
echo "You can now run the program using:"
echo ""
echo -e "${BLUE}  go run write_full_text_search.go \\${NC}"
echo -e "${BLUE}      -parquet-dir data/wikipedia/ \\${NC}"
echo -e "${BLUE}      -file-id-start 0 \\${NC}"
echo -e "${BLUE}      -file-id-end 10 \\${NC}"
echo -e "${BLUE}      -user-id 0${NC}"
echo ""
echo "Or build and run:"
echo ""
echo -e "${BLUE}  go build -o write_fts write_full_text_search.go${NC}"
echo -e "${BLUE}  ./write_fts -parquet-dir data/wikipedia/ -file-id-start 0 -file-id-end 10 -user-id 0${NC}"
echo ""
echo "For more options, run:"
echo -e "${BLUE}  go run write_full_text_search.go -help${NC}"
echo ""
