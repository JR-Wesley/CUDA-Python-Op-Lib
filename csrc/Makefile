CXX := nvcc
CXXFLAGS := -O2 -std=c++17 -Xcompiler -Wall \
			-gencode arch=compute_89,code=sm_89
# CXXFLAGS += -g -G

# Search all the files
KERNEL_DIR := kernel
TEST_DIR := test
BUILD_DIR := build/cpp

KERNEL_SRC := $(wildcard $(KERNEL_DIR)/*.cu)
KERNEL_OBJ := $(patsubst $(KERNEL_DIR)/%.cu, $(BUILD_DIR)/%.o, $(KERNEL_SRC))

TEST_SRC := $(wildcard $(TEST_DIR)/*.cpp)
TEST_NAM := $(patsubst $(TEST_DIR)/%.cpp, %, $(TEST_SRC))
TEST_TARGETS := $(patsubst %, $(BUILD_DIR)/%, $(TEST_NAM))

.PHONY: all clean dir

all: dir $(TEST_TARGETS)
	@echo "✅ All tests built in $(BUILD_DIR)/"

dir:
	@mkdir -p $(BUILD_DIR)

# 规则：将 .cu 编译为 .o（放在 build/ 下）
$(BUILD_DIR)/%.o: $(KERNEL_DIR)/%.cu
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# 规则：每个 test_*.cpp 生成一个可执行文件
$(BUILD_DIR)/%: $(TEST_DIR)/%.cpp $(KERNEL_OBJ)
	$(CXX) $(CXXFLAGS) $< $(KERNEL_OBJ) -o $@

clean:
	rm -rf $(BUILD_DIR)

info:
	@echo "KERNEL_SOURCES: $(KERNEL_SRC)"
	@echo "KERNEL_OBJS: $(KERNEL_OBJ)"
	@echo "TEST_SOURCES: $(TEST_SRC)"
	@echo "TEST_TARGETS: $(TEST_TARGETS)"

