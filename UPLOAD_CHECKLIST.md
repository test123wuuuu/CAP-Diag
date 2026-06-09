# 上传清单与匿名化报告

## ✅ 匿名化完成

### 已处理的匿名化项目

1. **论文** - 已经匿名（Anonymous Submission）
2. **代码路径** - 所有绝对路径已替换为相对路径
   - `D:/wuuuu/emnlp/` → `./data/`
   - 共清理了 7 个 Python 文件
3. **临时文件** - 已排除所有个人工作文件
4. **用户标识** - 没有发现个人邮箱或姓名

### 验证结果
- ✅ 代码中无绝对路径残留
- ✅ 论文使用匿名作者
- ✅ 无个人标识信息
- ✅ 目录结构清晰

---

## 📦 需要上传的内容

### 目录结构概览

```
anonymous-emnlp2026/                    (~600KB，不含大数据文件)
├── README.md                           # 项目概述
├── REPRODUCE.md                        # 复现指南
├── requirements.txt                    # Python 依赖
├── LICENSE                             # 开源许可
├── .gitignore                          # Git 忽略规则
│
├── paper/                              (~600KB)
│   ├── main.tex                        # 主论文源码
│   ├── main.pdf                        # PDF (445KB)
│   ├── acl.sty                         # ACL 样式
│   ├── acl_natbib.bst                  # BibTeX 样式
│   ├── tables/                         # 7个表格文件
│   │   ├── table_main_results.tex
│   │   ├── table_cross_mech.tex
│   │   ├── table_gate_sens.tex
│   │   ├── table_mi_vs_det.tex
│   │   ├── table_mi_robust.tex
│   │   ├── table_held_out.tex
│   │   └── table_related_work.tex
│   └── references/
│       └── custom.bib                  # 参考文献
│
├── code/                               (~100KB)
│   ├── protocols.py                    # 核心协议定义 (R0/R1/R2)
│   └── experiments/
│       ├── capacity/                   # 容量实验
│       │   ├── e3_generate_exp.py      # EXP 生成
│       │   ├── e3_capacity_exp.py      # EXP 容量
│       │   └── measure_capacity.py     # 容量度量
│       ├── triage/                     # 分流实验
│       │   └── e2_cap_triage_sim.py
│       ├── ablations/                  # 消融实验
│       │   └── phase_f_ablations.py
│       └── length_control/             # 长度控制
│           ├── round10_length_disentanglement.py
│           ├── round11_generate_r1short.py
│           └── round11_r1short_metrics.py
│
└── data/
    └── sample_inputs/                  # 数据样本
        ├── README.md                   # 数据说明
        └── sample_candidates.jsonl     # 10条样本记录
```

### 文件清单（共30个文件）

#### 根目录文档 (5个)
- README.md
- REPRODUCE.md  
- requirements.txt
- LICENSE
- .gitignore

#### 论文文件 (11个)
- paper/main.tex
- paper/main.pdf
- paper/acl.sty
- paper/acl_natbib.bst
- paper/references/custom.bib
- paper/tables/table_*.tex (7个表格)

#### 代码文件 (9个)
- code/protocols.py (核心)
- code/experiments/capacity/*.py (3个)
- code/experiments/triage/*.py (1个)
- code/experiments/ablations/*.py (1个)
- code/experiments/length_control/*.py (3个)

#### 数据文件 (2个)
- data/sample_inputs/README.md
- data/sample_inputs/sample_candidates.jsonl

---

## 🚫 已排除的内容（不上传）

### 1. 个人研究工具和配置
- ❌ `.claude/` - Claude Code 配置
- ❌ `Auto-claude-code-research-in-sleep/` - 研究助手工具
- ❌ `.aris/` - ARIS 系统配置

### 2. 早期探索和开发日志
- ❌ `idea-stage/`, `idea-stage-v2/` - idea 探索
- ❌ `refine-logs/`, `refine-logs-v2/` - 开发日志（~200个文件）
- ❌ `review-stage/`, `review-round10/` - 内部评审
- ❌ `round*/` - 各轮次迭代文件

### 3. 非最终实验
- ❌ `experiments/idea6/`, `experiments/idea6_r4/` - 其他 idea
- ❌ `experiments/idea5/runs/` - 运行结果（30GB）
- ❌ `experiments/idea5/data/*.jsonl` - 完整数据集（30GB）
- ❌ `experiments/idea5/models/` - 模型文件
- ❌ `experiments/idea5/cache/`, `logs/` - 缓存和日志

### 4. 图片生成过程
- ❌ `figure_outputs/` - 图片生成中间过程
- ❌ `figure_outputs_image_trials/` - 图片试验
- ❌ `draw/` - 手绘素材

### 5. 临时和会话文件
- ❌ `2026-05-*.txt` - 会话记录
- ❌ `审稿意见.txt` - 中文审稿意见
- ❌ `RESEARCH_BRIEF.md` - 个人研究笔记

---

## 📊 统计信息

- **上传文件数**: 30 个
- **总大小**: ~700KB（不含大型数据集）
- **论文**: 445KB PDF + 源码
- **代码**: 9 个核心 Python 脚本
- **排除内容**: ~30GB（实验数据+日志+临时文件）

---

## 🔍 最终检查清单

### 匿名性检查
- [x] 无作者姓名
- [x] 无机构信息
- [x] 无个人邮箱
- [x] 无绝对路径包含用户名
- [x] 论文标注为 "Anonymous Submission"

### 完整性检查
- [x] 论文 PDF 和源码完整
- [x] 核心实验代码包含
- [x] 依赖文件清晰
- [x] 复现指南详细
- [x] README 清晰

### 可编译性
- [x] LaTeX 可独立编译
- [x] Python 代码无语法错误
- [x] 路径引用为相对路径
- [x] 依赖版本明确

---

## 🚀 上传步骤

### 1. 初始化 Git 仓库

```bash
cd anonymous-emnlp2026
git init
git add .
git commit -m "Initial commit: CAP-Diag anonymous submission"
```

### 2. 创建 GitHub 匿名仓库

访问 https://github.com/new 创建新仓库：
- Repository name: `capdiag-emnlp2026` 或其他匿名名称
- Visibility: **Public**（或 Private，取决于会议要求）
- 不要添加 README/License（已包含在项目中）

### 3. 推送到 GitHub

```bash
git remote add origin https://github.com/YOUR_ANONYMOUS_ACCOUNT/capdiag-emnlp2026.git
git branch -M main
git push -u origin main
```

### 4. 验证上传

访问仓库 URL，检查：
- [ ] 目录结构正确
- [ ] README.md 正常显示
- [ ] PDF 可下载
- [ ] 代码可查看
- [ ] 无敏感信息泄露

### 5. 生成 Release（可选）

为投稿创建一个 release：
```bash
git tag -a v1.0-submission -m "EMNLP 2026 submission version"
git push origin v1.0-submission
```

---

## 📝 投稿时的仓库链接

在论文投稿系统中提供：

```
Code repository: https://github.com/YOUR_ANONYMOUS_ACCOUNT/capdiag-emnlp2026
Release: v1.0-submission
```

---

## ⚠️ 注意事项

1. **确保匿名账号**: 使用全新的 GitHub 账号，不关联个人信息
2. **检查 commit 信息**: git config 使用匿名邮箱
3. **避免泄露**: 不要在 Issues/Pull Requests 中透露身份
4. **数据可用性**: 在论文中说明完整数据集将在论文接收后公开
5. **投稿后不修改**: 创建 release 后，在评审期间不要修改代码

---

## 📧 后续操作

### 评审期间
- 监控 GitHub Issues（如果有审稿人提问）
- 准备好完整数据集，等待评审后公开

### 论文接收后
1. 更新仓库，移除匿名化限制
2. 上传完整数据集到 Hugging Face
3. 添加作者信息和机构
4. 更新 README 和 Citation
5. 创建正式 release

---

生成时间: 2026-06-09
状态: ✅ 准备就绪，可以上传
