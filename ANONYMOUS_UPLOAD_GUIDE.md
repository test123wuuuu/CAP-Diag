# 匿名提交操作指南（安全方案）

## ⚠️ 重要提醒

你的信息会泄露身份：
- ❌ GitHub账号：test123wuuuu
- ❌ 邮箱：wcy1536046459@126.com

**必须使用完全匿名的方式！**

---

## ✅ 推荐方案：anonymous.4open.science

### 步骤1：访问并创建仓库

1. 打开浏览器，访问：https://anonymous.4open.science

2. 点击右上角 **"Sign in with GitHub"** 或 **"Create Repository"**

3. 如果需要登录：
   - 可以用任何GitHub账号（包括你的test123wuuuu）
   - **但创建的仓库是匿名的**，审稿人看不到你的账号

4. 创建新仓库：
   - Repository name: `capdiag-emnlp2026`
   - 选择 **Anonymous** 模式
   - 不要初始化README

5. 复制生成的URL（会是类似）：
   ```
   https://anonymous.4open.science/r/capdiag-emnlp2026-ABCD
   ```

### 步骤2：推送代码

```bash
cd D:\wuuuu\emnlp\anonymous-emnlp2026

# 更新远程地址（替换为你实际得到的URL）
git remote remove origin
git remote add origin https://anonymous.4open.science/r/capdiag-emnlp2026-ABCD

# 推送
git push -u origin main
```

### 步骤3：验证

访问你的匿名URL，确认：
- ✅ 文件都在
- ✅ 没有显示你的GitHub账号
- ✅ PDF可以下载

---

## 备选方案：创建新的匿名GitHub账号

如果anonymous.4open.science有问题，可以：

### 步骤1：创建临时邮箱

访问以下任一服务：
- https://temp-mail.org
- https://10minutemail.com
- https://guerrillamail.com

获取临时邮箱（如 `random123@tempmail.com`）

### 步骤2：注册匿名GitHub账号

1. 访问 https://github.com/signup
2. 使用临时邮箱注册
3. 用户名：`anonymous-emnlp-2026` 或 `capdiag-submission`
4. 不要添加任何个人信息

### 步骤3：创建仓库并推送

```bash
cd D:\wuuuu\emnlp\anonymous-emnlp2026

git remote remove origin
git remote add origin https://github.com/anonymous-emnlp-2026/capdiag.git
git push -u origin main
```

---

## ❌ 不要做的事

1. ❌ 不要用 test123wuuuu 账号公开上传
2. ❌ 不要在commit中包含真实邮箱
3. ❌ 不要在仓库描述中透露学校/机构
4. ❌ 不要在Issues/Discussions中使用真名

---

## 📝 投稿时填写

使用匿名URL：

```
Code Repository: https://anonymous.4open.science/r/YOUR-REPO-ID
或
Code Repository: https://github.com/anonymous-emnlp-2026/capdiag

Status: Anonymous for double-blind review
License: MIT
```

---

## 🔄 论文接收后

可以将匿名仓库转换为正式仓库：
1. anonymous.4open.science 支持一键转换到GitHub
2. 或者重新推送到你的实名账号

---

## 下一步

**请选择一个方案告诉我**：

A. 使用 anonymous.4open.science（最推荐）
B. 创建新的匿名GitHub账号
C. 其他方案

我会继续协助你完成！
