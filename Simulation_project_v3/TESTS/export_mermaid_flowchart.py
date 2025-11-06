#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mermaid流程图导出工具

将Mermaid图表转换为高质量图片（PNG/SVG）
"""

import re
import base64
import requests
from pathlib import Path
from typing import List, Dict


def extract_mermaid_code(md_file: Path) -> List[Dict[str, str]]:
    """
    从Markdown文件中提取Mermaid代码块
    
    Args:
        md_file: Markdown文件路径
    
    Returns:
        Mermaid代码块列表
    """
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 正则匹配Mermaid代码块
    pattern = r'```mermaid\n(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL)
    
    charts = []
    for idx, code in enumerate(matches, 1):
        charts.append({
            'index': idx,
            'code': code.strip(),
            'title': f'chart_{idx}'
        })
    
    return charts


def export_via_kroki(mermaid_code: str, output_path: Path, format: str = 'png'):
    """
    使用Kroki服务导出Mermaid图表
    
    Args:
        mermaid_code: Mermaid代码
        output_path: 输出文件路径
        format: 输出格式（png/svg/pdf）
    """
    # Kroki公共API
    kroki_url = f"https://kroki.io/mermaid/{format}"
    
    # 编码Mermaid代码
    encoded = base64.urlsafe_b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
    
    # 发送请求
    url = f"{kroki_url}/{encoded}"
    
    try:
        print(f"  正在请求Kroki服务...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # 保存文件
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"  ✓ 成功导出: {output_path}")
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"  ✗ 导出失败: {e}")
        return False


def export_via_mermaid_ink(mermaid_code: str, output_path: Path):
    """
    使用mermaid.ink服务导出PNG图表
    
    Args:
        mermaid_code: Mermaid代码
        output_path: 输出文件路径
    """
    # mermaid.ink公共API
    base_url = "https://mermaid.ink/img/"
    
    # 编码Mermaid代码
    encoded = base64.urlsafe_b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
    url = f"{base_url}{encoded}"
    
    try:
        print(f"  正在请求mermaid.ink服务...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # 保存文件
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"  ✓ 成功导出: {output_path}")
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"  ✗ 导出失败: {e}")
        return False


def generate_html_preview(mermaid_code: str, output_path: Path):
    """
    生成包含Mermaid图表的HTML预览文件
    
    Args:
        mermaid_code: Mermaid代码
        output_path: 输出HTML文件路径
    """
    html_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gale-Shapley算法流程图</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            themeVariables: {{
                fontSize: '16px',
                fontFamily: 'Arial, sans-serif'
            }}
        }});
    </script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }}
        .mermaid {{
            text-align: center;
            background-color: white;
        }}
        .instructions {{
            margin-top: 30px;
            padding: 15px;
            background-color: #e8f4f8;
            border-left: 4px solid #2196F3;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Gale-Shapley稳定匹配算法流程图</h1>
        
        <div class="mermaid">
{mermaid_code}
        </div>
        
        <div class="instructions">
            <h3>💡 使用说明</h3>
            <p><strong>导出图片：</strong> 右键点击图表，选择"Save image as..."保存为PNG</p>
            <p><strong>缩放查看：</strong> 使用浏览器的缩放功能（Ctrl +/-）调整大小</p>
            <p><strong>打印输出：</strong> 按Ctrl+P打印或保存为PDF</p>
        </div>
    </div>
</body>
</html>"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"  ✓ HTML预览文件已生成: {output_path}")


def main():
    """主函数"""
    print("="*60)
    print("Mermaid流程图导出工具")
    print("="*60)
    
    # 输入文件
    md_file = Path("MODULES/VISUALIZATION/gs_algorithm_flowchart.md")
    
    if not md_file.exists():
        print(f"✗ 错误: 找不到文件 {md_file}")
        return
    
    # 提取Mermaid代码
    print(f"\n从 {md_file} 提取Mermaid图表...")
    charts = extract_mermaid_code(md_file)
    print(f"✓ 找到 {len(charts)} 个图表")
    
    # 输出目录
    output_dir = Path("OUTPUT/flowcharts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 导出每个图表
    for chart in charts:
        idx = chart['index']
        code = chart['code']
        
        print(f"\n处理图表 {idx}...")
        
        # 确定文件名
        if idx == 1:
            name = "gs_algorithm_complete"
        elif idx == 2:
            name = "gs_algorithm_simplified"
        else:
            name = f"gs_algorithm_chart_{idx}"
        
        # 方法1: 使用Kroki导出PNG
        print(f"方法1: 使用Kroki服务导出PNG...")
        png_path = output_dir / f"{name}.png"
        export_via_kroki(code, png_path, 'png')
        
        # 方法2: 使用Kroki导出SVG
        print(f"方法2: 使用Kroki服务导出SVG...")
        svg_path = output_dir / f"{name}.svg"
        export_via_kroki(code, svg_path, 'svg')
        
        # 方法3: 使用mermaid.ink导出PNG（备用）
        print(f"方法3: 使用mermaid.ink服务导出PNG（备用）...")
        png_ink_path = output_dir / f"{name}_ink.png"
        export_via_mermaid_ink(code, png_ink_path)
        
        # 生成HTML预览
        print(f"生成HTML预览文件...")
        html_path = output_dir / f"{name}.html"
        generate_html_preview(code, html_path)
    
    print("\n" + "="*60)
    print("✓ 所有图表导出完成！")
    print(f"✓ 输出目录: {output_dir}")
    print("="*60)
    
    print("\n📌 提示:")
    print("1. PNG图片可以直接用于Word/PPT")
    print("2. SVG矢量图适合学术出版（可缩放不失真）")
    print("3. HTML文件可在浏览器中打开预览")
    print("4. 如果在线服务失败，请检查网络连接或稍后重试")
    print("\n💡 本地导出方案（需要Node.js）:")
    print("   npm install -g @mermaid-js/mermaid-cli")
    print("   mmdc -i MODULES/VISUALIZATION/gs_algorithm_flowchart.md -o output.png")


if __name__ == '__main__':
    main()


