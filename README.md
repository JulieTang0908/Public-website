# 一维随机排列：可重新模拟的后验概率曲面

公开网页：https://julietang0908.github.io/Public-website/

A=[1] 占一格，B=[2,2] 占两格，空格为 0；合法组成满足 a+2b≤N。合法数量组合等概率，给定组成的合法摆放等概率。组成后验对所有摆放的高斯似然取平均，并在对数空间用动态规划求和。二值模式保留潜在长杆，但将其观测读数变为 [1,1]。

## 网页使用

- 默认包括 N=1～100 每个整数，可设置起点/终点，或只看当前 N。
- 选择模式、σ、概率/区间密度，旋转曲面或查看热图和二维切片。
- 首次打开使用600条已核验的参考曲线；修改 σ 会按需计算缺少的曲线。
- 展开“重新模拟一组观测”，输入随机种子、基本序列，再点击重新模拟。后台工作线程在本设备求后验，无账号、API或服务器依赖。
- 新模拟使用 xorshift32 + Box-Muller，Python 与 JavaScript 对同一种子给出一致的标准噪声。旧参考保留归档的 NumPy 噪声，因此即使填写旧种子也不要求新模拟与历史向量相同。
- 改变种子或基本序列时会清空旧后验缓存，避免不同观测混在同一曲面。不同 N、模式和 σ 共享同一标准噪声向量以便对照，并非独立准确率实验。
- 当前后验可下载 JSON，含完整观测、候选组成、概率和真值。新计算还保留 log_probability；极小概率在最终转换时可能舍入为零。归档数据没有保存对数概率的地方导出 null，不补造数值。
- F0/F2 边际分布来自两层均匀先验，不依赖某一次模拟图片；此视图使用已保存的总体计算。后验曲面不能称为 P(F|N)。

## 命令行重新生成

网页使用不需要安装依赖。如需批量生成数据，Python 3.10+ 安装 NumPy 与 SciPy。在完整工作区中运行：

```powershell
python -m part2.main_goal_surface.simulate --config part2/main_goal_surface/config.json --output part2/main_goal_surface/output/simulation
```

GitHub 仓库里的独立源码位于 src；下载后可直接运行：

```powershell
python -m pip install -r src/requirements.txt
python src/simulate.py --config src/config.json --output results
python src/simulate.py --N-min 20 --N-max 50 --sigma 0.3 --mode binary --seed 12345 --skip-features --output results/custom
```

配置文件控制 N 范围、σ 列表、观测模式列表、种子、基本序列、F2 抽样数和精确枚举的最大 N。命令行选项覆盖对应配置项。枚举阈值限制在 0～10，避免意外枚举大视野。

如果已有逐格连续读数，保存为 `{"z":[0.2,1.1,1.8,2.2]}`，执行：

```powershell
python src/simulate.py --observations observation.json --sigma 0.5 --mode typed --skip-features --output results/observed
```

这会推断一个实际输入的格子向量，真值为 null；不包含照片分割或网格标定。JSON/CSV 是独立实验导出，不会自动覆盖网页当前数据；网页重新模拟入口用于交互计算。

输出为 simulation.json 和 posterior.csv。每行记录 N、模式、σ、组成编号 x、u、a、b、p、log p、归一化编号的区间密度。JSON 同时保存配置摘要、观测和特征边际。F0 全部精确；F2 默认 N≤8 精确，其余每个 N 按两层先验抽样30,000次，可改配置。输出是模拟结果，不附加未经验证的正态或准确率结论。

## 数学与读图

```text
K_N = (floor(N/2)+1)(N+1-floor(N/2))
x = b(N+2-b)+a; u = x/(K_N-1)
R(N,a,b) = C(N-b,b) C(N-2b,a)
P(a,b|z,N) ∝ [sum_M exp(-||z-M||²/(2σ²))]/R(N,a,b)
sum_(a,b) P(a,b|z,N) = 1
归一化编号的区间密度 = (K_N-1)*p
```

x 是可逆组成编号，不是物理距离或长杆比例。密度可大于 1。三角面只连接离散数据。均匀先验的编号分布水平；原始 F0/F2 混合分布另看，Z 标准化不会强制正态。σ 为测量误差标准差，不是错误百分比。

## 代码和检查

- engine.js：浏览器/Node 共用的滚动层对数动态规划、编号、生成器。
- live_controls.js：后台计算、范围选择、缓存隔离和 JSON 导出。
- model.py：Python 模型、相同随机生成器、精确/抽样特征边际。
- simulate.py、config.json：批量计算与配置。
- page.html、build_standalone.py：网页模板与独立包组装；后者复用工作区 output 中已保存的参考数据和绘图库。
- test_engine.py：小 N 完整枚举、600条归档曲线、额外噪声尺度、编号与随机生成器的交叉核对。

工作区检查：`python -m unittest part2.main_goal_surface.test_engine -v`。

组装网页：`python -m part2.main_goal_surface.build_standalone`。输出为 standalone/dist/index.html 与 standalone/composition-probability-standalone.zip；ZIP根目录可直接交给静态托管平台。模型计算源码也随网页发布在 src。
