# 推送到 anonymous.4open.science 指南

## ✅ 当前状态

Git仓库已准备就绪：
- ✅ Git已初始化
- ✅ 所有文件已提交（30个文件）
- ✅ 远程仓库已配置：`https://anonymous.4open.science/r/CAP-Diag-3E5E`
- ✅ 匿名身份已设置

---

## 🚀 推送步骤

### 方式1：直接推送（如果你已有访问权限）

```bash
cd /d/wuuuu/emnlp/anonymous-emnlp2026
git push -u origin main
```

### 方式2：如果需要认证

4open.science 通常需要通过网站生成访问令牌：

1. **访问你的仓库页面**：
   https://anonymous.4open.science/r/CAP-Diag-3E5E

2. **获取推送令牌**：
   - 在页面上查找 "Settings" 或 "Access Token"
   - 生成一个推送令牌

3. **使用令牌推送**：
   ```bash
   git push https://TOKEN@anonymous.4open.science/r/CAP-Diag-3E5E main
   ```

### 方式3：如果仓库是你创建的

如果这个URL是你刚创建的空仓库：

```bash
# 推送到main分支
git push -u origin main

# 或推送到master分支（如果4open要求）
git push -u origin master
```

---

## 🔍 验证推送成功

推送后，访问：
https://anonymous.4open.science/r/CAP-Diag-3E5E

应该能看到：
- ✅ README.md 显示
- ✅ paper/main.pdf 可下载
- ✅ 目录结构正确
- ✅ 所有30个文件

---

## ⚠️ 常见问题

### 问题1：推送被拒绝（Permission denied）

**解决方案**：
1. 确认你在4open.science上有这个仓库的写权限
2. 检查是否需要访问令牌
3. 确认URL是否正确

### 问题2：仓库已存在内容

**解决方案**：
```bash
# 先拉取远程内容
git pull origin main --allow-unrelated-histories

# 解决冲突后推送
git push origin main
```

### 问题3：分支名称不匹配

**解决方案**：
```bash
# 检查远程分支名
git branch -r

# 推送到对应分支
git push origin main:main
# 或
git push origin master:master
```

---

## 📝 论文投稿时填写

在EMNLP 2026投稿系统中填写：

```
Code Repository: https://anonymous.4open.science/r/CAP-Diag-3E5E
Reproducibility: See REPRODUCE.md in repository
License: MIT
```

---

## 🎯 下一步

1. **现在推送**：
   ```bash
   git push -u origin main
   ```

2. **验证**：访问 https://anonymous.4open.science/r/CAP-Diag-3E5E

3. **完成投稿**：在论文系统中提供仓库链接

---

## 📞 如果遇到问题

- 检查4open.science的文档
- 确认仓库是否已经在你的账号下创建
- 如果是空仓库，直接推送即可
- 如果需要合并，先pull再push

---

准备就绪！运行 `git push -u origin main` 开始推送。
