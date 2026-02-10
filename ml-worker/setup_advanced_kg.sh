#!/bin/bash
# setup_advanced_kg.sh - Install all advanced knowledge graph dependencies

echo "🚀 Installing Advanced Knowledge Graph Dependencies"
echo "=================================================="

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment active: $VIRTUAL_ENV"
else
    echo "⚠️  WARNING: Not in a virtual environment!"
    echo "   Please activate your venv first: source venv/bin/activate"
    exit 1
fi

echo ""
echo "Step 1: Installing PyTorch (if not already installed)"
echo "----------------------------------------------------"
# Check if PyTorch is already installed
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo "✅ PyTorch already installed: $TORCH_VERSION"
else
    echo "Installing PyTorch..."
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "Step 2: Installing PyTorch Geometric and dependencies"
echo "----------------------------------------------------"
# Get PyTorch version for compatibility
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "2.1.0")
TORCH_MAJOR=$(echo $TORCH_VERSION | cut -d. -f1,2)

echo "Installing PyG for PyTorch $TORCH_MAJOR..."

# Install PyG dependencies in order
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH_MAJOR}.0+cpu.html
pip install torch-geometric==2.4.0

echo ""
echo "Step 3: Installing Transformer models for NER"
echo "----------------------------------------------------"
pip install transformers==4.46.0

echo ""
echo "Step 4: Installing NetworkX for graph analysis"
echo "----------------------------------------------------"
pip install networkx==3.2.1

echo ""
echo "Step 5: Verifying installations"
echo "----------------------------------------------------"

# Test imports
python - << 'PYEOF'
import sys

print("\n🧪 Testing imports...")
errors = []

# Test PyTorch
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"❌ PyTorch: {e}")
    errors.append("torch")

# Test PyTorch Geometric
try:
    import torch_geometric
    print(f"✅ PyTorch Geometric {torch_geometric.__version__}")
except ImportError as e:
    print(f"❌ PyTorch Geometric: {e}")
    errors.append("torch_geometric")

# Test PyG components
try:
    from torch_geometric.nn import GCNConv, GATConv
    print(f"✅ PyTorch Geometric layers (GCNConv, GATConv)")
except ImportError as e:
    print(f"❌ PyTorch Geometric layers: {e}")
    errors.append("torch_geometric.nn")

# Test torch-scatter
try:
    import torch_scatter
    print(f"✅ torch-scatter")
except ImportError as e:
    print(f"❌ torch-scatter: {e}")
    errors.append("torch_scatter")

# Test torch-sparse
try:
    import torch_sparse
    print(f"✅ torch-sparse")
except ImportError as e:
    print(f"❌ torch-sparse: {e}")
    errors.append("torch_sparse")

# Test Transformers
try:
    import transformers
    print(f"✅ Transformers {transformers.__version__}")
except ImportError as e:
    print(f"❌ Transformers: {e}")
    errors.append("transformers")

# Test NetworkX
try:
    import networkx
    print(f"✅ NetworkX {networkx.__version__}")
except ImportError as e:
    print(f"❌ NetworkX: {e}")
    errors.append("networkx")

if errors:
    print(f"\n❌ Some packages failed to import: {', '.join(errors)}")
    sys.exit(1)
else:
    print("\n✅ All packages installed successfully!")
    sys.exit(0)
PYEOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ Installation completed successfully!"
    echo "=================================================="
    echo ""
    echo "You can now run your worker:"
    echo "  python src/worker.py"
else
    echo ""
    echo "=================================================="
    echo "❌ Installation had errors"
    echo "=================================================="
    echo ""
    echo "Try manual installation:"
    echo "  pip install torch torchvision torchaudio"
    echo "  pip install torch-scatter torch-sparse torch-geometric"
    echo "  pip install transformers networkx"
    exit 1
fi