# AutoDL GPU 训练指南

本指南将帮助你在 AutoDL 平台上使用 GPU 进行 PGPortfolio 模型训练。

## 一、AutoDL 平台准备

### 1.1 租用 GPU 实例
1. 登录 [AutoDL](https://www.autodl.com/)
2. 选择适合的 GPU 实例（推荐 RTX 3090/4090 或更高配置）
3. 选择系统镜像：**Ubuntu 20.04** 或 **Ubuntu 22.04**（推荐）
4. 启动实例

### 1.2 上传代码
AutoDL 提供了多种上传方式：

**方式一：使用 JupyterLab 上传**
- 在 AutoDL 控制台点击 "JupyterLab"
- 在文件浏览器中上传整个项目文件夹

**方式二：使用 Git 克隆**
```bash
# 在 AutoDL 终端中执行
cd /root/autodl-tmp  # 或你的工作目录
git clone <your-repo-url>
cd PGPortfolio-master
```

**方式三：使用 AutoDL 的文件传输工具**
- 在控制台使用 "文件传输" 功能上传压缩包
- 在终端解压：
```bash
cd /root/autodl-tmp
unzip PGPortfolio-master.zip
cd PGPortfolio-master
```

## 二、环境配置

### 2.1 检查 GPU 可用性
```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 CUDA 版本
nvcc --version
```

### 2.2 安装 Python 依赖
AutoDL 通常已预装 Python 3.8+，但需要安装项目依赖：

```bash
# 进入项目目录
cd /root/autodl-tmp/PGPortfolio-master  # 根据实际路径调整

# 创建虚拟环境（可选但推荐）
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 如果 PyTorch 版本不匹配，可能需要单独安装支持 CUDA 的版本
# 例如：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2.3 验证 PyTorch GPU 支持
```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

如果输出显示 `CUDA available: True`，说明 GPU 配置成功。

## 三、数据准备

### 3.1 上传数据库文件
将你的 `database/okx_data.db` 文件上传到 AutoDL 实例：

```bash
# 确保数据库目录存在
mkdir -p database

# 如果使用 JupyterLab，直接拖拽上传到 database/ 目录
# 或使用 scp 从本地传输：
# scp database/okx_data.db root@<autodl-ip>:/root/autodl-tmp/PGPortfolio-master/database/
```

### 3.2 检查配置文件
确保 `pgportfolio/net_config.json` 中的数据库路径正确：
```json
{
  "input": {
    "database_file": "okx_data.db"
  }
}
```

## 四、开始训练

### 4.1 使用 GPU 训练
```bash
# 基本训练命令（使用 GPU）
python main.py --mode train --device gpu

# 如果使用多个进程（不推荐，除非有多个 GPU）
python main.py --mode train --device gpu --processes 1
```

### 4.2 后台训练（推荐）
由于训练时间较长，建议使用 `nohup` 或 `screen` 在后台运行：

**方式一：使用 nohup**
```bash
nohup python main.py --mode train --device gpu > train.log 2>&1 &
```

**方式二：使用 screen（推荐）**
```bash
# 安装 screen（如果未安装）
apt-get update && apt-get install -y screen

# 创建新的 screen 会话
screen -S pgportfolio_train

# 在 screen 中运行训练
python main.py --mode train --device gpu

# 按 Ctrl+A 然后按 D 来分离会话（训练继续运行）
# 重新连接会话：screen -r pgportfolio_train
```

### 4.3 监控训练进度
```bash
# 查看训练日志
tail -f train_package/1/programlog

# 或者如果使用 nohup
tail -f train.log

# 监控 GPU 使用情况
watch -n 1 nvidia-smi
```

## 五、常见问题排查

### 5.1 GPU 未被使用
如果训练时 GPU 使用率为 0%，检查：

1. **确认设备参数正确**：
   ```bash
   python main.py --mode train --device gpu  # 注意是 "gpu" 不是 "cuda"
   ```

2. **检查代码中的设备检测**：
   ```python
   # 在代码中，device == "gpu" 才会使用 CUDA
   # 查看 pgportfolio/learn/nnagent.py 第19行
   ```

3. **验证 PyTorch CUDA 支持**：
   ```python
   import torch
   print(torch.cuda.is_available())  # 应该输出 True
   ```

### 5.2 显存不足（OOM）
如果遇到 `CUDA out of memory` 错误：

1. **减小批次大小**：
   编辑 `pgportfolio/net_config.json`：
   ```json
   {
     "training": {
       "batch_size": 64  // 从 128 减小到 64 或更小
     }
   }
   ```

2. **使用梯度累积**（需要修改代码）

3. **选择显存更大的 GPU 实例**

### 5.3 训练中断
AutoDL 实例可能会因为网络问题或超时中断，建议：

1. **使用 screen 或 tmux** 保持会话
2. **定期保存检查点**（如果代码支持）
3. **使用 AutoDL 的持久化存储**保存重要文件

## 六、性能优化建议

### 6.1 批次大小调整
根据 GPU 显存调整：
- RTX 3090 (24GB): batch_size = 128-256
- RTX 4090 (24GB): batch_size = 128-256
- RTX 3060 (12GB): batch_size = 64-128
- 较小显存: batch_size = 32-64

### 6.2 混合精度训练（可选）
如果代码支持，可以启用混合精度训练以加速：
```python
# 需要在代码中添加
from torch.cuda.amp import autocast, GradScaler
```

### 6.3 数据加载优化
确保数据加载不会成为瓶颈，可以：
- 使用多进程数据加载（如果 DataLoader 支持）
- 将数据预处理缓存到内存

## 七、下载训练结果

训练完成后，下载模型和结果：

1. **通过 JupyterLab**：直接在文件浏览器中下载 `train_package/` 目录
2. **使用 scp**：
   ```bash
   # 在本地终端执行
   scp -r root@<autodl-ip>:/root/autodl-tmp/PGPortfolio-master/train_package ./
   ```

## 八、快速开始脚本

创建一个快速启动脚本 `train_gpu.sh`：

```bash
#!/bin/bash
# train_gpu.sh

echo "Checking GPU availability..."
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo "Starting training with GPU..."
python main.py --mode train --device gpu

echo "Training completed!"
```

使用：
```bash
chmod +x train_gpu.sh
./train_gpu.sh
```

## 九、AutoDL 特殊配置

### 9.1 持久化存储
AutoDL 提供持久化存储，建议将重要数据放在：
```bash
# 通常路径为
/root/autodl-tmp  # 临时存储（实例停止后可能丢失）
# 或使用持久化存储挂载点（根据 AutoDL 文档配置）
```

### 9.2 自动启动训练
创建 `~/.bashrc` 或启动脚本，在实例启动时自动开始训练（可选）

## 十、验证 GPU 训练

训练开始后，可以通过以下方式验证 GPU 正在使用：

```bash
# 实时监控 GPU
watch -n 1 nvidia-smi

# 应该看到：
# - GPU 使用率 > 0%
# - 显存使用量增加
# - Python 进程在 GPU 上运行
```

---

**注意事项**：
- AutoDL 按使用时长计费，训练完成后及时停止实例
- 重要文件及时下载，避免数据丢失
- 建议先用小数据集测试，确认 GPU 正常工作后再进行完整训练

