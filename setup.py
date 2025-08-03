"""
For release, use `pip install -e` to compile all the operators into a package
"""
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from pathlib import Path

def find_cuda_extensions():
    extensions = []
    binding_dir = Path("src") / "bindings"
    kernel_dir = Path("src") / "kernels"

    if not binding_dir.exists():
        return []

    for cpp_file in binding_dir.glob("*.cpp"):
        op_name = cpp_file.stem
        sources = [str(cpp_file)]
        cu_file = kernel_dir / f"{op_name}.cu"
        if cu_file.exists():
            sources.append(str(cu_file))

        print(f"Build: {op_name} from {sources}")

        extensions.append(
            CUDAExtension(
                name=f"cuda_ops.{op_name}",
                sources=sources,
                include_dirs=[str(Path("src") / "utils")],
                extra_compile_args={
                    'cxx': ['-O3'],
                    'nvcc': ['-O3', '--use_fast_math']
                },
                define_macros=[('TORCH_EXTENSION_NAME', f'cuda_ops_{op_name}')]
            )
        )
    return extensions

setup(
    name="cuda_ops",
    version="0.1.0",
    description="Custom CUDA Operators for PyTorch",
    author="Your Name",
    packages=find_packages(),
    ext_modules=find_cuda_extensions(),
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
