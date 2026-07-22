#!/bin/bash
# FlexKV 测试运行脚本
# 自动设置必要的库路径并运行测试

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve the active PyTorch runtime library directory instead of relying on a
# machine-specific CUDA stub path. This works for both CUDA and ROCm PyTorch.
TORCH_LIB_DIR="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')" || exit $?
export LD_LIBRARY_PATH="${SCRIPT_DIR}/build/lib:${TORCH_LIB_DIR}:${LD_LIBRARY_PATH}"

echo "=========================================="
echo "FlexKV 测试运行脚本"
echo "=========================================="
echo "库路径已设置:"
echo "  - ${SCRIPT_DIR}/build/lib"
echo "  - ${TORCH_LIB_DIR}"
echo ""

# 检查是否提供了测试文件参数
if [ -z "$1" ]; then
    echo "运行默认测试: tests/test_dis_radixtree_basic.py"
    python3 "${SCRIPT_DIR}/tests/test_dis_radixtree_basic.py"
else
    echo "运行测试: $1"
    python3 "$@"
fi

exit_code=$?
echo ""
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "✓ 测试完成 (退出码: $exit_code)"
else
    echo "✗ 测试失败 (退出码: $exit_code)"
fi
echo "=========================================="

exit $exit_code

