# 🚀 CUDA Ops Template：PyTorch 自定义 CUDA 算子开发模板

一个简洁、可扩展、开箱即用的 PyTorch CUDA 扩展开发模板，专为高效开发多个自定义算子（如 `matmul`、`bitonic_sort`、`MoE` 等）而设计。

- ✅ 支持 **开发期即时编译**（无需安装）
- ✅ 支持 **发布期 pip 安装**（便于集成）
- ✅ 支持增加 **自定义算子**
- ✅ 使用 `LibTorch + PyBind11` 编写 C++ 接口
<!-- - ✅ 使用 `torch.utils.cpp_extension` 替代 CMake，简化构建 -->
- ✅ 包含测试与性能基准脚本

---

## 📁 项目结构

```bash
cuda-ops-template/
├── src/
│   ├── kernels/              # CUDA kernel 实现（.cu 文件）
│   │   └── matmul.cu           # 示例：矩阵乘法 kernel
│   ├── bindings/             # C++ 绑定层（.cpp 文件）
│   │   └── matmul.cpp          # 示例：PyBind11 接口
│   └── utils/                # 通用头文件
│       ├── common.h          # CHECK_CUDA, CHECK_CONTIGUOUS 等宏
├── tests/                    # 单元测试脚本
│   └── test_matmul.py
├── benchmarks/               # 性能测试脚本
│   └── benchmark_matmul.py
├── compile_dev.py            # 开发模式：即时编译 + 自动加载
├── setup.py                  # 发布模式：支持 pip install
├── pyproject.toml            # Python 包元数据（可选）
└── README.md                 # 本文件
```

### 特点

csrc 现代 C++ 风格 tests

TODO: libtorch + pybind11 封装 Python API


### 算子

Elemwise:
- Add
Reduce:

Matmul:


---

## TODO: ⚙️ 编译与使用方式

### Make

### 开发模式：即时编译（推荐用于开发调试）

无需安装，修改代码后重新运行自动重新编译。

```bash
# 安装依赖（首次）
pip install torch>=2.0 pybind11

# 运行编译脚本（自动发现所有算子）
python compile_dev.py
```

运行后会输出：
```
🔍 发现算子: matmul -> ['src/bindings/matmul.cpp', 'src/kernels/matmul.cu']
✅ matmul 编译成功
可用算子: ['matmul']
```

#### 在 Python 中使用：

```python
import torch
from compile_dev import cuda_ops  # 动态加载的算子集合

x = torch.randn(100, 200, device='cuda')
w = torch.randn(200, 300, device='cuda')

# 调用自定义 CUDA 算子
y = cuda_ops['matmul'].matmul_forward(x, w)
print(y.shape)  # torch.Size([100, 300])
```

---

### 2. 发布模式：pip 安装（推荐用于模型集成）

将算子编译为 Python 包，支持 `import cuda_ops.xxx`。

```bash
# 安装为可导入的包（-e 表示可编辑模式）
pip install -e .
```

安装成功后，在任意 Python 脚本中使用：

```python
import torch
import cuda_ops.matmul

x = torch.randn(100, 200, device='cuda')
w = torch.randn(200, 300, device='cuda')

y = cuda_ops.matmul.matmul_forward(x, w)
```

> 💡 适用于训练脚本、模型部署、CI/CD 流程。

---

## ✅ 测试算子正确性

```bash
python tests/test_matmul.py
```

预期输出：
```
✅ matmul 测试通过
```

---

## 📊 性能基准测试

```bash
python benchmarks/benchmark_matmul.py
```

预期输出：
```
🚀 自定义 matmul 耗时: 1.2345 ms/次
```

---

## ➕ 如何添加新算子？（例如 `gelu` 或 `bitonic_sort`）

只需两步，无需修改任何构建脚本！

### 步骤 1：创建 CUDA Kernel
```bash
# 例如添加 GELU 近似计算
touch src/kernels/gelu.cu
```

### 步骤 2：创建 C++ 绑定
```bash
touch src/bindings/gelu.cpp
```

> 文件名必须一致（如 `xxx.cu` + `xxx.cpp`）

### 步骤 3：重新编译
```bash
# 开发模式
python compile_dev.py

# 或发布模式
pip install -e .
```

即可使用：
```python
# 开发模式
y = cuda_ops['gelu'].gelu_forward(x)

# 发布模式
import cuda_ops.gelu
y = cuda_ops.gelu.gelu_forward(x)
```

---

## Dependencies

- Python >= 3.8
- PyTorch >= 2.0（需 CUDA 版本匹配）
- CUDA Toolkit（与 PyTorch 版本对应）
- `pybind11`（通常由 PyTorch 自动提供）


### Environment

It's recommended using `uv` to install dependencies.

```bash
# 1. 创建环境
uv venv Path-to-venv
source Path-to-venv/bin/activate

# 2. 安装依赖
uv pip install -r requirements.txt

# 3. 安装你的 CUDA 扩展
uv pip install -e .

# 4. 验证
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
"
```

---

## 📚 参考资料

- PyTorch CUDA Extensions: https://pytorch.org/docs/stable/cpp_extension.html
- PyBind11: https://pybind11.readthedocs.io/
- cuBLAS: https://docs.nvidia.com/cuda/cublas/

