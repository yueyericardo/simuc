import urllib.parse
import re


def convert(text, addtoc=False):
    text = convert_latex(text)
    if addtoc:
        header, text = generate_toc(text)
        text = "## Table of content\n\n" + header + '\n' + text
    return text


def convert_latex(text):
    def toimage(x):
        if x[1] == r'$' and x[-2] == r'$':
            x = x[2:-2]
            img = "\n<img src='https://math.now.sh?from={}' style='display: block; margin: 0.5em auto;'>\n"
            return img.format(urllib.parse.quote_plus(x))
        else:
            x = x[1:-1]
            img = "<img src='https://math.now.sh?from={}' style='display: inline-block; margin: 0;'>"
            return img.format(urllib.parse.quote_plus(x))
    return re.sub(r'\${2}([^$]+)\${2}|\$(.+?)\$', lambda x: toimage(x.group()), text)


def generate_toc(mdtext, start_level=2, end_level=6):
    """
    Code modified from https://github.com/waynerv/github-markdown-toc/blob/master/gfm_toc/md_toc.py
    """

    if start_level > end_level:
        return
    output_md = ""
    anchor_tracker = {}
    block_flag = False
    # Compile regular expression objects
    # 编译需要重复使用的正则表达式对象
    regex_title = re.compile(r'^(#+) (.*)$')
    regex_tag = re.compile(r'<.*?>')
    regex_char = re.compile(r'[~`!@#$%^&*(){}\[\];:\"\'<,.>?\/\\|_+=-]')
    regex_space = re.compile(r' ')
    regex_block = re.compile(r'^\s*```')
    mdtext = mdtext.splitlines()

    for i, process_line in enumerate(mdtext):
        # Check if the text is inside the code block
        # 检查文本是否在代码块内
        if regex_block.match(process_line):
            if block_flag:
                block_flag = False
            else:
                block_flag = True
        # If the text is in the code block, skip the header matching and subsequent processing of that line.
        # 若文本在代码块内，则跳过该行文本的标题匹配与后续处理
        if block_flag:
            continue
        else:
            # Perform a regular match on current line, returning a match object
            # 对该行文本执行标题的正则匹配，返回一个match object
            match = regex_title.search(process_line)

            if match:
                header_level = len(match.group(1))

                # Skip headers that are not within the specified header level
                # 跳过不在指定标题级别范围内的标题
                if header_level < start_level or header_level > end_level:
                    continue

                # Delete the HTML tag in the text
                # 删除标题文本中的HTML标签
                header_text = regex_tag.sub('', match.group(2))
                # Generate header anchor, convert text to lowercase, and remove spaces or extra symbols from text
                # 生成锚点，将标题文本转换为小写，并删除标题文本中的空格与多余符号
                header_anchor = regex_char.sub('', (header_text.lower()))
                header_anchor = regex_space.sub('-', header_anchor)

                # Refer to the GFM specification to process the same heading anchor
                # 参照GFM规范，对转换后文本相同的标题锚点进行处理
                if header_anchor not in anchor_tracker:
                    anchor_tracker[header_anchor] = 0
                else:
                    anchor_tracker[header_anchor] += 1
                    header_anchor = header_anchor + '-' + str(anchor_tracker[header_anchor])

                # Indented space multiple of the output result
                # 输出结果的缩进空格倍数
                indents = header_level - start_level

                # Output the formatted TOC entry, such as: - [Use FlaskForm-Migrate](#use-flaskform-migrate)
                # 输出格式化后的目录条目，如：- [Use FlaskForm-Migrate](#use-flaskform-migrate)
                output_md += " " * (indents * 2) + "- [" + header_text + "](#" + header_anchor + ")\n"
                # replace all header with html header
                mdtext[i] = "<h{} id='{}'>{}</h{}>".format(header_level, header_anchor, header_text, header_level)

    return output_md, '\n'.join(mdtext)
