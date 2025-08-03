"""
For development: JIT compile source code in `src/kernels/` and `src/bindings/` 
"""
from torch.utils.cpp_extension import load
import os
from pathlib import Path

ROOT = Path(__file__).parent

def discover_extensions():
    extensions = {}
    binding_dir = ROOT / "src" / "bindings"
    kernel_dir = ROOT / "src" / "kernels"

    if not binding_dir.exists():
        print("bindings/ 目录不存在")
        return {}

    for cpp_file in binding_dir.glob("*.cpp"):
        op_name = cpp_file.stem
        cu_file = kernel_dir / f"{op_name}.cu"
        sources = [str(cpp_file)]
        if cu_file.exists():
            sources.append(str(cu_file))

        print(f"Operator: {op_name} -> {sources}")

        try:
            extensions[op_name] = load(
                name=f"cuda_ops_{op_name}",
                sources=sources,
                extra_include_paths=[str(ROOT / "src" / "utils")],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                with_cuda=True,
                verbose=False
            )
            print(f"{op_name} build succeed")
        except Exception as e:
            print(f"{op_name} build failed: {e}")

    return extensions

# Load all the operators
cuda_ops = discover_extensions()

if __name__ == "__main__":
    print("\nAll the operators:", list(cuda_ops.keys()))

