# High-Dimensional BO with Task Adapters (1220 Version)

本项目基于 BOAD 框架（High-Dimensional Bayesian Optimization with Additive Structure Learning），新增了对多种实际高维任务（Rover Trajectory, MIP, NAS, Lasso/SVM）的原生支持。所有任务通过统一的 `Task Adapter` 接口进行管理，支持在离线或受限网络环境下运行。

## 1. 环境安装 (Installation)

由于服务器可能无法访问 GitHub，建议使用清华源进行安装。

### 1.1 基础依赖
在项目根目录下运行：
```bash
pip install -r requirements.txt -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)
1.2 任务特定依赖 (按需安装)
A. NAS-Bench-201 (用于 NAS 架构搜索任务)

Bash

pip install nas-bench-201 -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)
重要提示：代码库安装后，必须手动下载数据库文件 NAS-Bench-201-v1_1-096897.pth (约 2GB)，并上传到项目的 data/ 目录下。

B. PySCIPOpt (用于 MIP 求解任务) MIP 任务依赖 SCIP 求解器。请确保系统层已安装 SCIP Suite，然后安装 Python 接口：

Bash

pip install pyscipopt -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)
降级机制：如果未安装 PySCIPOpt，MIP 任务将自动运行在 Mock 模式（使用模拟函数），仅用于流程测试，不会报错。

C. LassoBench (用于 SVM/DNA 任务) 本项目内置了基于 scikit-learn 的原生实现适配器，无需安装 lassobench 第三方库。只要安装了 scikit-learn 即可运行。

2. 数据准备 (Data Preparation)
请确保项目根目录下存在 data/ 文件夹，并按需放置以下数据文件：

Plaintext

project_root/
├── data/
│   ├── NAS-Bench-201-v1_1-096897.pth  # [必须] 运行 NAS 任务需要
│   ├── dna.csv                        # [可选] 运行 DNA 任务 (如缺失将使用合成数据)
│   └── mip_instances/                 # [必须] 运行 MIP 真实求解需要
│       ├── qiu.mps
│       └── misc05.mps
3. 运行实验 (Running Experiments)
所有实验均通过入口脚本 scripts/run_experiment.py 运行，只需指定对应的配置文件。

A. 运行 Rover 轨迹规划 (60D)
描述: 60维机器人路径规划，具有几何序列依赖结构。

数据: 无需额外数据（原生 NumPy 实现）。

Bash

python scripts/run_experiment.py --config configs/task_rover.yaml
B. 运行 SVM 特征选择 (388D)
描述: 基于 Breast Cancer 数据集的高维稀疏特征选择（模拟 SVM 参数调优）。

数据: 使用 sklearn 内置数据集，无需额外下载。

Bash

python scripts/run_experiment.py --config configs/task_svm.yaml
C. 运行 MIP 求解 (74D)
描述: 混合整数规划参数调优（Instance: Qiu）。

数据: 需要 data/mip_instances/qiu.mps。

Bash

python scripts/run_experiment.py --config configs/task_mip.yaml
如果未安装 SCIP，将输出 [MIPTask] PySCIPOpt not found. Running in MOCK mode. 并继续运行。

D. 运行 NAS 架构搜索 (30D)
描述: 在 NAS-Bench-201 搜索空间中寻找最优神经网络架构。

数据: 需要 data/NAS-Bench-201-v1_1-096897.pth。

Bash

python scripts/run_experiment.py --config configs/task_nas.yaml
4. 配置文件指南 (Configuration Guide)
在 configs/ 目录下创建或修改 .yaml 文件来控制实验参数。

示例 1：Rover 任务配置
YAML

experiment:
  name: "Rover_60D_Test"
  seed: 42
  device: "cuda"

problem:
  type: "rover"        # 指定任务类型为 rover
  name: "rover_60d"
  dim: 60              # Rover 任务固定为 60维
  
optimization:
  n_initial: 50
  n_total: 300
  switch_threshold: 50 # 第50轮后开启自适应结构学习

algorithm:
  decomposition_method: "friedman" # 推荐使用 friedman 发现几何结构
  decomp_freq: 25
示例 2：MIP 任务配置
YAML

problem:
  type: "mip"
  name: "mip_qiu"
  dim: 74
  task_config:
    instance: "qiu"       # 对应 data/mip_instances/qiu.mps
    time_limit: 10.0      # SCIP 求解限时
示例 3：Lasso/SVM 任务配置
YAML

problem:
  type: "lasso"        # Lasso, SVM, DNA 统称 lasso 类型
  name: "svm_388d"
  dim: 388
  task_config:
    dataset: "svm"     # 选项: 'svm' (388D) 或 'dna' (180D)
5. 常见问题 (FAQ)
Q1: 为什么日志中显示的 Regret 是 None？ A: 对于实际黑盒任务（如 MIP, NAS, SVM），我们通常无法得知理论上的全局最优值（Global Optimum）。因此无法计算 Regret。请关注日志中的 y_best (Best Observed Value)，该值越大越好（代码内部已将最小化问题转换为最大化）。

Q2: 如何添加新的任务？ A: 本项目采用了 Task Adapter 模式。

在 src/bo_core/tasks/ 下新建任务文件（继承 BaseTask）。

在 src/bo_core/tasks/__init__.py 的工厂方法中注册该任务。

创建对应的 YAML 配置文件。

Q3: 运行 NAS 任务时报错 FileNotFoundError？ A: 请检查 YAML 配置文件中 task_config.data_path 指向的路径是否正确，并确认是否已手动上传了 .pth 数据库文件。
