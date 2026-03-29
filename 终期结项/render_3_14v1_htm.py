"""将 3.14v1 Markdown 主稿渲染为适合复制到 Word 的 HTM 文件。"""

from __future__ import annotations

import html
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_ROOT / "论文" / "3.14v1.md"
OUTPUT_PATH = PROJECT_ROOT / "论文" / "3.14v1_排版版.htm"

BODY_STYLE = (
    "margin:0;padding:0;font-family:'Times New Roman',serif;"
)
PARA_STYLE = (
    "font-family:宋体,'Times New Roman',serif;font-size:10.5pt;"
    "line-height:28pt;text-indent:21pt;margin-top:0pt;margin-bottom:0pt;"
    "text-align:justify;"
)
NOTE_STYLE = (
    "font-family:宋体,'Times New Roman',serif;font-size:9.5pt;"
    "line-height:22pt;text-indent:0;margin-top:4pt;margin-bottom:4pt;"
    "text-align:left;"
)
H1_STYLE = (
    "font-family:黑体,'Times New Roman',serif;font-size:16pt;font-weight:bold;"
    "text-align:center;line-height:28pt;margin-top:28pt;margin-bottom:28pt;"
)
H2_STYLE = (
    "font-family:黑体,'Times New Roman',serif;font-size:14pt;font-weight:bold;"
    "text-align:left;line-height:28pt;margin-top:28pt;margin-bottom:28pt;"
)
CAPTION_STYLE = (
    "font-family:宋体,'Times New Roman',serif;font-size:10.5pt;font-weight:bold;"
    "text-align:center;line-height:20pt;margin-top:8pt;margin-bottom:4pt;"
)
IMAGE_WRAP_STYLE = "text-align:center;margin-top:4pt;margin-bottom:8pt;"
IMAGE_STYLE = (
    "max-width:440pt;width:auto;height:auto;"
)
EQUATION_STYLE = (
    "font-family:'Times New Roman',serif;font-size:10.5pt;line-height:28pt;"
    "text-align:center;margin-top:6pt;margin-bottom:6pt;"
)
TABLE_CELL_STYLE = (
    "padding:3pt 5pt;font-family:宋体,'Times New Roman',serif;"
    "font-size:9.5pt;text-align:center;vertical-align:middle;"
)
ALGO_BOX_STYLE = (
    "width:440pt;border-collapse:collapse;border:1.5pt solid black;"
    "margin-top:4pt;margin-bottom:8pt;"
)
ALGO_TITLE_STYLE = (
    "padding:4pt 8pt;border-bottom:1.5pt solid black;"
    "font-family:宋体,'Times New Roman',serif;font-size:10pt;font-weight:bold;"
)
ALGO_IO_STYLE = (
    "padding:3pt 8pt;border-bottom:0.5pt solid black;"
    "font-family:宋体,'Times New Roman',serif;font-size:10pt;"
)
ALGO_CODE_STYLE = (
    "padding:4pt 8pt;font-family:'Courier New',monospace;font-size:9.5pt;"
    "line-height:160%;"
)


def compact_whitespace(text: str) -> str:
    """压缩公式中的多余空白，避免 Word 解析时受到换行干扰。"""

    return re.sub(r"\s+", " ", text).strip()


def extract_equation_number(content: str) -> tuple[str, str]:
    """从行间公式末尾提取编号，并将编号移出公式主体。"""

    patterns = [
        r"\\#\((\d+)\)\s*$",
        r"\\#\\left\.\s*[（(](\d+)[）)]\s*\\right\.\s*$",
        r"\\#\s*[（(](\d+)[）)]\s*$",
    ]
    cleaned = content
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            number = match.group(1)
            cleaned = re.sub(pattern, "", cleaned).strip()
            return cleaned, f"({number})"
    return cleaned, ""


def sanitize_formula_content(content: str, display_mode: bool) -> tuple[str, str]:
    """将公式清洗为更接近 Word 线性公式可识别的形式。"""

    cleaned = content.strip()
    cleaned = re.sub(r"\\begin\{array\}\{[^}]+\}", "", cleaned)
    cleaned = re.sub(r"\\end\{array\}", "", cleaned)
    cleaned = cleaned.replace(r"\lbrack", "[")
    cleaned = cleaned.replace(r"\rbrack", "]")
    cleaned = cleaned.replace(r"\text{\{}", r"\{")
    cleaned = cleaned.replace(r"\text{\}}", r"\}")
    cleaned = cleaned.replace(r"\text{|}", r"|")
    cleaned = cleaned.replace(r"\text{/}", "/")
    cleaned = cleaned.replace(r"\text{\{}}", r"\{")
    cleaned = cleaned.replace(r"\text{\}}", r"\}")
    cleaned = cleaned.replace(r"\mathbb{R}^{\mathbb{4}}", r"\mathbb{R}^{4}")
    cleaned = cleaned.replace(r"\xi_{i}^{\ }", r"\xi_i")
    cleaned = cleaned.replace(r"\theta_{t} = V\text{/}U_{t}", r"\theta_t = V/U_t")
    cleaned = cleaned.replace(r"\theta^{(k + 1)} \leftarrow V\text{/}U^{\text{new}}", r"\theta^{(k+1)} \leftarrow V/U^{\mathrm{new}}")
    cleaned = cleaned.replace(r"U^{\text{new}}", r"U^{\mathrm{new}}")
    cleaned = cleaned.replace(r"m^{\text{new}}", r"m^{\mathrm{new}}")
    cleaned = cleaned.replace(r"\text{diag}", r"\mathrm{diag}")
    cleaned = cleaned.replace(r"\text{|}", "|")
    cleaned = cleaned.replace(r"\text{\{}(i,j)\text{\}}", r"\{(i,j)\}")
    cleaned = cleaned.replace(r"\text{\{}0,1\text{\}}", r"\{0,1\}")
    cleaned = cleaned.replace(r"\arg{\min_", r"\arg\min_")
    cleaned = cleaned.replace(r"\arg{\max_", r"\arg\max_")
    cleaned = cleaned.replace(r"\text{\{}}\cdots\text{\}}", r"\{\cdots\}")
    cleaned = re.sub(r"\\right(\]|\)|\\\})\s*}", r"\\right\1", cleaned)
    cleaned = cleaned.replace(r"\left. ", "")
    cleaned = cleaned.replace(r" \right.", "")
    cleaned = compact_whitespace(cleaned)

    if display_mode:
        cleaned, equation_number = extract_equation_number(cleaned)
    else:
        equation_number = ""

    cleaned = cleaned.replace("( ", "(").replace(" )", ")")
    cleaned = cleaned.replace("[ ", "[").replace(" ]", "]")
    cleaned = cleaned.replace("{ ", "{").replace(" }", "}")
    cleaned = cleaned.replace("}'", "'")
    return cleaned, equation_number


def sanitize_formula_token(token: str) -> str:
    """清洗被 $ 或 $$ 包裹的公式文本。"""

    if token.startswith("$$") and token.endswith("$$"):
        inner = token[2:-2]
        cleaned, _ = sanitize_formula_content(inner, display_mode=True)
        return f"$${cleaned}$$"

    if token.startswith("$") and token.endswith("$"):
        inner = token[1:-1]
        cleaned, _ = sanitize_formula_content(inner, display_mode=False)
        return f"${cleaned}$"

    return token


def protect_special_text(text: str) -> tuple[str, dict[str, str]]:
    """保护需要原样保留的片段，避免被后续转义和正则替换破坏。"""

    placeholders: dict[str, str] = {}

    def reserve(pattern: str, source: str) -> str:
        def repl(match: re.Match[str]) -> str:
            key = f"__TOKEN_{len(placeholders)}__"
            placeholders[key] = sanitize_formula_token(match.group(0))
            return key

        return re.sub(pattern, repl, source, flags=re.DOTALL)

    protected = text
    protected = reserve(r"\$\$.*?\$\$", protected)
    protected = reserve(r"\$.*?\$", protected)
    protected = protected.replace(r"\*", "__ESCAPED_STAR__")
    return protected, placeholders


def restore_special_text(text: str, placeholders: dict[str, str]) -> str:
    """恢复被保护的数学公式和转义符号。"""

    restored = text.replace("__ESCAPED_STAR__", "*")
    for key, value in placeholders.items():
        restored = restored.replace(key, value)
    return restored


def convert_inline(text: str) -> str:
    """将行内 Markdown 转换为 HTML，同时保留 LaTeX 原文。"""

    protected, placeholders = protect_special_text(text)
    escaped = html.escape(protected)
    escaped = re.sub(r"\*\*\*(.+?)\*\*\*", r"<b><i>\1</i></b>", escaped)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", escaped)
    restored = restore_special_text(escaped, placeholders)
    return re.sub(r"\$\$(.+?)\$\$", r"$\1$", restored)


def convert_preserving_spaces(text: str) -> str:
    """用于算法步骤，保留行首缩进和行内格式。"""

    leading = len(text) - len(text.lstrip(" "))
    converted = convert_inline(text.lstrip(" "))
    return "&nbsp;" * leading + converted


def render_heading(level: int, text: str) -> str:
    style = H1_STYLE if level == 1 else H2_STYLE
    tag = "h1" if level == 1 else "h2"
    return f'    <{tag} style="{style}">{convert_inline(text.strip())}</{tag}>'


def render_paragraph(text: str) -> str:
    style = NOTE_STYLE if text.startswith("注：") else PARA_STYLE
    return f'    <p style="{style}">{convert_inline(text.strip())}</p>'


def render_caption(text: str) -> str:
    return f'    <p style="{CAPTION_STYLE}">{convert_inline(text.strip())}</p>'


def render_image(image_path: str) -> str:
    src = html.escape(image_path, quote=True)
    return (
        f'    <p style="{IMAGE_WRAP_STYLE}">'
        f'<img src="{src}" style="{IMAGE_STYLE}" alt="插图"></p>'
    )


def render_equation(lines: list[str]) -> str:
    raw_text = "\n".join(lines)
    inner = raw_text.strip()
    if inner.startswith("$$"):
        inner = inner[2:]
    if inner.endswith("$$"):
        inner = inner[:-2]
    cleaned, equation_number = sanitize_formula_content(inner, display_mode=True)
    equation_html = html.escape(f"$${cleaned}$$", quote=False)
    equation_html = equation_html.replace("&#x27;", "'")
    if equation_number:
        equation_html = (
            f"{equation_html}&nbsp;&nbsp;{html.escape(equation_number, quote=False)}"
        )
    return f'    <p style="{EQUATION_STYLE}">{equation_html}</p>'


def infer_table_width(rows: list[list[str]]) -> str:
    column_count = max((len(row) for row in rows), default=0)
    return "440pt" if column_count >= 4 else "auto"


def render_markdown_table(rows: list[list[str]]) -> str:
    """渲染普通三线表，尽量保持 Word 可识别的简洁结构。"""

    content_rows = rows[:]
    if len(content_rows) >= 2 and all(
        set(cell.replace(" ", "")) <= {"-", ":"} for cell in content_rows[1]
    ):
        content_rows.pop(1)

    table_width = infer_table_width(content_rows)
    html_lines = [
        (
            f'    <table align="center" style="width:{table_width};'
            "border-collapse:collapse;border-top:2pt solid black;"
            'border-bottom:2pt solid black;margin-top:4pt;margin-bottom:8pt;">'
        )
    ]

    for row_index, row in enumerate(content_rows):
        clean_cells = [cell.strip() for cell in row]
        non_empty = [cell for cell in clean_cells if cell]

        if row_index == 0:
            html_lines.append('        <tr style="border-bottom:1pt solid black;">')
            cell_tag = "td"
            cell_style = TABLE_CELL_STYLE.replace(
                "text-align:center;",
                "font-weight:bold;text-align:center;",
            )
        elif len(non_empty) == 1 and len(clean_cells) > 1:
            html_lines.append("        <tr>")
            html_lines.append(
                "            "
                f'<td colspan="{len(clean_cells)}" style="padding:3pt 5pt;'
                "font-family:宋体,'Times New Roman',serif;font-size:9.5pt;"
                f'font-style:italic;text-align:left;">{convert_inline(non_empty[0])}</td>'
            )
            html_lines.append("        </tr>")
            continue
        else:
            html_lines.append("        <tr>")
            cell_tag = "td"
            cell_style = TABLE_CELL_STYLE

        for cell in clean_cells:
            html_lines.append(
                "            "
                f"<{cell_tag} style=\"{cell_style}\">{convert_inline(cell)}</{cell_tag}>"
            )
        html_lines.append("        </tr>")

    html_lines.append("    </table>")
    return "\n".join(html_lines)


def split_markdown_row(line: str) -> list[str]:
    """按管道分割表格行，并去掉两侧空白单元。"""

    parts = [part.strip() for part in line.strip().strip("|").split("|")]
    return parts


def render_algorithm(title: str, input_line: str, output_line: str, steps: list[str]) -> str:
    step_html = "<br>\n".join(convert_preserving_spaces(line) for line in steps)
    return "\n".join(
        [
            render_caption(title),
            f'    <table align="center" style="{ALGO_BOX_STYLE}">',
            "        <tr>",
            f'            <td style="{ALGO_TITLE_STYLE}">{convert_inline(title)}</td>',
            "        </tr>",
            "        <tr>",
            f'            <td style="{ALGO_IO_STYLE}">{convert_inline(input_line)}</td>',
            "        </tr>",
            "        <tr>",
            f'            <td style="{ALGO_IO_STYLE}">{convert_inline(output_line)}</td>',
            "        </tr>",
            "        <tr>",
            f'            <td style="{ALGO_CODE_STYLE}">{step_html}</td>',
            "        </tr>",
            "    </table>",
        ]
    )


def read_next_non_empty(lines: list[str], start_index: int) -> tuple[str, int]:
    """读取下一个非空行，并返回该行内容及其下一位置。"""

    index = start_index
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines):
        return "", index
    return lines[index].strip(), index + 1


def is_algorithm_line(raw_line: str) -> bool:
    """判断当前行是否仍属于算法步骤区域。"""

    stripped = raw_line.strip()
    if not stripped:
        return True
    if re.match(r"^\d+\.", stripped):
        return True
    if raw_line.startswith(("   ", "\t")):
        return True
    return False


def parse_markdown(markdown_text: str) -> list[str]:
    """按块解析 Markdown，只覆盖当前主稿实际使用到的语法。"""

    lines = markdown_text.splitlines()
    result: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        if stripped.startswith("<!--") and stripped.endswith("-->"):
            result.append(f"    {stripped}")
            index += 1
            continue

        if stripped.startswith("# "):
            result.append(render_heading(1, stripped[2:]))
            index += 1
            continue

        if stripped.startswith("## "):
            result.append(render_heading(2, stripped[3:]))
            index += 1
            continue

        if stripped.startswith("$$"):
            equation_lines = [line]
            index += 1
            while index < len(lines):
                equation_lines.append(lines[index])
                if lines[index].strip().endswith("$$"):
                    index += 1
                    break
                index += 1
            result.append(render_equation(equation_lines))
            continue

        if stripped.startswith("![") and "](" in stripped and stripped.endswith(")"):
            image_path = stripped[stripped.index("](") + 2 : -1]
            result.append(render_image(image_path))
            index += 1
            continue

        if stripped.startswith("|"):
            table_lines = [stripped]
            index += 1
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            table_rows = [split_markdown_row(table_line) for table_line in table_lines]
            result.append(render_markdown_table(table_rows))
            continue

        if re.fullmatch(r"\*\*算法.+\*\*", stripped):
            title = stripped.strip("*")
            input_line, next_index = read_next_non_empty(lines, index + 1)
            output_line, next_index = read_next_non_empty(lines, next_index)
            step_lines: list[str] = []
            index = next_index
            while index < len(lines):
                current = lines[index]
                if not is_algorithm_line(current):
                    break
                if current.strip():
                    step_lines.append(current.rstrip())
                index += 1
            result.append(render_algorithm(title, input_line, output_line, step_lines))
            continue

        if stripped.startswith("**表") or stripped.startswith("**图"):
            result.append(render_caption(stripped))
            index += 1
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            current = lines[index].strip()
            if not current:
                break
            if current.startswith(("# ", "## ", "|", "![", "<!--", "$$")):
                break
            if current.startswith("**表") or current.startswith("**图") or re.fullmatch(
                r"\*\*算法.+\*\*", current
            ):
                break
            paragraph_lines.append(current)
            index += 1
        result.append(render_paragraph(" ".join(paragraph_lines)))

    return result


def build_document(body_blocks: list[str]) -> str:
    """组装完整 HTM 文档。"""

    body = "\n".join(body_blocks)
    return "\n".join(
        [
            "<!DOCTYPE html>",
            "<html>",
            "",
            "<head>",
            '    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">',
            "    <title>3.14v1 排版版</title>",
            "</head>",
            "",
            f'<body style="{BODY_STYLE}">',
            "",
            "    <!-- 本文件由 render_3_14v1_htm.py 自动生成，适用于复制到 Word。 -->",
            "    <!-- 全部样式采用行内样式，数学公式保留原始 LaTeX 写法。 -->",
            "",
            body,
            "",
            "</body>",
            "",
            "</html>",
            "",
        ]
    )


def main() -> None:
    """读取 Markdown 主稿并输出 HTM 文件。"""

    markdown_text = INPUT_PATH.read_text(encoding="utf-8")
    html_text = build_document(parse_markdown(markdown_text))
    OUTPUT_PATH.write_text(html_text, encoding="utf-8")
    print(f"已生成：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
