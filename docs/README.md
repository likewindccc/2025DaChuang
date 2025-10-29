# 静态网站说明

## 📁 文件结构

```
WEBSITE/
├── index.html              # 首页（已完成）
├── mfg.html               # MFG模块页面（待创建）
├── calibration.html       # 校准模块页面（待创建）
├── simulation.html        # 政策仿真页面（待创建）
├── about.html             # 关于项目页面（待创建）
└── figures/               # Plotly HTML图表文件夹
    ├── mfg/              # MFG模块图表
    ├── calibration/      # 校准模块图表
    └── simulation/       # 仿真模块图表
```

## 🎨 网站特点

- ✅ **响应式设计**：自适应电脑/平板/手机
- ✅ **Bootstrap框架**：美观现代的UI
- ✅ **交互式图表**：Plotly集成，支持缩放、悬停、筛选
- ✅ **渐变配色**：紫色系专业配色
- ✅ **卡片悬浮效果**：增强交互体验

## 🚀 本地预览

直接双击 `index.html` 即可在浏览器中打开。

或使用Python启动本地服务器：

```bash
cd WEBSITE
python -m http.server 8000
# 访问 http://localhost:8000
```

## 📤 部署到GitHub Pages

### 步骤1：创建GitHub仓库

```bash
# 初始化仓库
git init
git add .
git commit -m "初始化MFG可视化网站"
```

### 步骤2：推送到GitHub

```bash
# 关联远程仓库（替换为你的GitHub用户名）
git remote add origin https://github.com/你的用户名/mfg-visualization.git
git branch -M main
git push -u origin main
```

### 步骤3：启用GitHub Pages

1. 进入仓库的 `Settings`
2. 找到 `Pages` 选项
3. Source 选择 `main` 分支
4. 文件夹选择 `/` (root)
5. 点击 `Save`

等待1-2分钟，访问：
```
https://你的用户名.github.io/mfg-visualization/
```

## 📊 添加Plotly图表

### 方法1：生成单独的HTML文件

```python
import plotly.graph_objects as go

fig = go.Figure(...)
fig.write_html('WEBSITE/figures/mfg/value_function.html')
```

然后在页面中嵌入：

```html
<iframe src="figures/mfg/value_function.html" 
        width="100%" height="600px" 
        frameborder="0">
</iframe>
```

### 方法2：直接嵌入到页面

```python
import plotly.graph_objects as go

fig = go.Figure(...)

# 获取HTML div代码
html_div = fig.to_html(include_plotlyjs='cdn', div_id='my-plot')

# 复制到HTML文件的<div>标签中
```

## 🎯 下一步

1. 创建其他页面（mfg.html, calibration.html等）
2. 生成Plotly图表并放入figures/文件夹
3. 在对应页面中嵌入图表
4. 测试所有链接和交互功能
5. 部署到GitHub Pages

## 📝 自定义配置

### 修改配色

在HTML的`<style>`标签中修改：

```css
/* 主色调（紫色渐变） */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* 可以改为其他颜色，如蓝色 */
background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
```

### 修改字体

```css
body {
    font-family: 'Microsoft YaHei', Arial, sans-serif;
}
```

### 添加logo

在导航栏的`<a class="navbar-brand">`中添加：

```html
<a class="navbar-brand" href="index.html">
    <img src="logo.png" height="30"> MFG可视化系统
</a>
```

## ⚠️ 注意事项

1. **文件路径**：确保所有链接使用相对路径
2. **图表大小**：Plotly图表文件可能较大（几MB），注意加载速度
3. **浏览器兼容**：推荐Chrome/Edge/Firefox最新版本
4. **CDN依赖**：Bootstrap和Plotly使用CDN，需要网络连接

## 📞 技术支持

如有问题，请查看：
- [Bootstrap文档](https://getbootstrap.com/docs/5.3/)
- [Plotly Python文档](https://plotly.com/python/)
- [GitHub Pages文档](https://docs.github.com/pages)

