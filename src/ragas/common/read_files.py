# read_files.py

import os
import re
import zipfile
import requests
import docx
import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """返回文本字符串中的Token数量"""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def unzip_file(zip_file_path, encoding='gbk'):
    # 定义解压目标目录
    extract_to_path = os.path.dirname(zip_file_path)
    extract_path = zip_file_path.rsplit('.', 1)[0]
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # 获取 ZIP 文件内的所有文件名
        zip_info_list = zip_ref.infolist()
        for zip_info in zip_info_list:
            # 将文件名重新编码，以正确处理中文字符
            zip_info.filename = zip_info.filename.encode('cp437').decode(encoding)
            # 解压单个文件
            zip_ref.extract(zip_info, extract_to_path)

    return extract_path

def list_files_in_folder(folder_path):
    """
    读取指定文件夹中的所有文件，返回文件路径列表。
    """
    file_paths = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        # 检查文件是否是普通文件
        if os.path.isfile(filepath):
            file_paths.append(filepath)

    return file_paths


def read_file(file_path):
    """
    根据文件路径读取文件后缀名、去掉后缀的文件名和文件内容，并返回一个包含这些信息的字典。
    如果无法成功读取文件，返回一个空列表。
    """
    result = {}

    # 提取文件后缀名和文件名
    file_name = os.path.basename(file_path)
    file_name_without_extension, file_extension = os.path.splitext(file_name)

    try:
        # 首先尝试使用 UTF-8 编码读取
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
    except UnicodeDecodeError:
        # 如果 UTF-8 读取失败，尝试使用 GBK 编码
        try:
            with open(file_path, 'r', encoding='gbk') as file:
                file_content = file.read()
        except Exception:
            # 如果读取仍然失败，返回空列表
            return []
    except Exception:
        # 如果出现其他异常，也返回空列表
        return []

    # 将后缀名、去掉后缀的文件名和文件内容添加到结果字典中
    result['file_extension'] = file_extension
    result['file_name'] = file_name
    result['file_content'] = file_content

    return result


def extract_md_title(md_content):
    """
    从Markdown内容中提取标题。
    
    参数:
    md_content (str): Markdown文件的内容。
    
    返回:
    str: 提取到的标题，如果提取失败则返回''。
    """
    # 匹配Markdown标题的正则表达式
    title_pattern = re.compile(r'^\s*#* (.+?)\s*$', re.MULTILINE)

    # 在内容中查找匹配的标题
    match = title_pattern.search(md_content)

    # 如果找到匹配的标题，返回提取到的标题；否则返回空字符串
    if match:
        return match.group(1)
    else:
        return ''
    

def split_markdown(text):
    # 临时替换论文引用和作者信息
    citation_pattern = re.compile(r'\[\d+\]')
    citations = citation_pattern.findall(text)
    for i, citation in enumerate(citations):
        text = text.replace(citation, f'__CITATION_{i}__')

    author_pattern = re.compile(r'\[@[^\]]+\]')
    authors = author_pattern.findall(text)
    for i, author in enumerate(authors):
        text = text.replace(author, f'__AUTHOR_{i}__')

    # 临时替换表格内容
    table_pattern = re.compile(r'\|.*\|')
    tables = table_pattern.findall(text)
    for i, table in enumerate(tables):
        text = text.replace(table, f'__TABLE_{i}__')

    # 将换行符替换为空格
    text = text.replace('\n', ' ')

    # 使用正则表达式找到句子结束符号（句号、问号、感叹号）
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # 还原论文引用和作者信息
    for i, citation in enumerate(citations):
        sentences = [sentence.replace(f'__CITATION_{i}__', citation) for sentence in sentences]

    for i, author in enumerate(authors):
        sentences = [sentence.replace(f'__AUTHOR_{i}__', author) for sentence in sentences]

    # 还原表格内容
    for i, table in enumerate(tables):
        sentences = [sentence.replace(f'__TABLE_{i}__', table) for sentence in sentences]

    return sentences


# def split_markdown_by_headings(markdown_text):
#     # 使用正则表达式按照标题拆分Markdown文本
#     headings = re.split(r'\n\s*#', markdown_text.strip())

#     # 在每个标题前添加#（除了第一个标题）
#     paragraphs = ['#' + heading if i > 0 else heading for i, heading in enumerate(headings)]

#     return paragraphs

# def split_docs_by_headings(text):
#     # 定义标题的正则表达式模式（这里假设标题格式为“数字. 标题”）
#     title_pattern = re.compile(r'^[一二三四五六七八九十]+、')

#     sections = []
#     current_section = []
#     current_title = None

#     for line in text.split('\n'):
#         # 检查当前行是否为新的大标题
#         match = title_pattern.match(line.strip())
#         if match:
#             # 如果当前部分不为空，保存并开始一个新部分
#             if current_section:
#                 sections.append('\n'.join(current_section))
#                 current_section = []
#         current_section.append(line.strip())
#         # 提取标题
#         current_title = match.group(0)
#     # 添加最后一部分
#     if current_section:
#         sections.append((current_title,'\n'.join(current_section)))

#     return sections

def split_markdown_by_headings(markdown_text):
    # 使用正则表达式按照标题拆分Markdown文本
    headings = re.split(r'\n\s*#', markdown_text.strip())
    print('headings',headings)

    # 在每个标题前添加#（除了第一个标题）
    paragraphs = ['#' + heading if i > 0 else heading for i, heading in enumerate(headings)]

    # 处理每个段落，提取标题级别和内容
    result = []
    current_title_level1,current_title_level2,current_title_level3 = None,None,None
    for paragraph in paragraphs:
        if paragraph=='' or paragraph is None:
            continue
        # print('paragraph',paragraph)
        
        # 使用正则表达式匹配标题级别和内容
        match = re.match(r'(#*)(.*)', paragraph)
        title_level = len(match.group(1))
        print('title_level',title_level)
        title_content = match.group(2).strip()
        print('title_content',title_content)
        current_title_level1 = title_content if title_level == 1 else current_title_level1 if title_level > 1 and current_title_level1 else None
        current_title_level2 = None if title_level < 2 else title_content if title_level == 2 else current_title_level2 if title_level > 2 and current_title_level2 else None
        current_title_level3 = None if title_level < 3 else title_content if title_level == 3 else current_title_level3 if title_level > 3 and current_title_level3 else None
        # print('match``',match)
        for section in paragraph.split('\n'):
            if section=='' or section is None:
                continue
            # 构建字典并添加到结果列表
            paragraph_dict = {
                'title_level1': current_title_level1,
                'title_level2': current_title_level2,
                'title_level3': current_title_level3,
                'content': section
            }
            result.append(paragraph_dict)
    return result

def split_docs_by_headings(text):
    # 定义标题的正则表达式模式
    title_pattern_1 = re.compile(r'^([一二三四五六七八九十]+)、(.*)')
    title_pattern_2 = re.compile(r'^(\d+)\.(.*)')

    result = []
    current_title_level1 = None
    current_title_level2 = None
    for section in text.split('\n'):
        # 检查当前行是否为新的大标题
        match_1 = title_pattern_1.match(section.strip())
        match_2 = title_pattern_2.match(section.strip())
        if section=='' or section is None:
            continue
        if match_1:
            # print('match_1',match_1)
            current_title_level1 = match_1.group(2)
            current_title_level2 = None
        elif match_2:
            current_title_level2 = match_2.group(2)

        paragraph_dict = {
                        'title_level1': current_title_level1,
                        'title_level2': current_title_level2,
                        'content': section
                    }
        result.append(paragraph_dict)
    return result


def split_paragraph_into_sentences(paragraph):
    # 使用中文句子结束符进行分割
    sentences = re.split(r'(?<=[。？！])', paragraph)
    # 移除空字符串
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences

def process_markdown(markdown_text, file_name):
    # 调用函数将Markdown拆分成段落
    paragraphs = split_markdown_by_headings(markdown_text)
    title = extract_md_title(markdown_text)
    num_tokens = num_tokens_from_string(markdown_text, "cl100k_base")

    # 构建结果列表
    result_list = []
    for idx, paragraph in enumerate(paragraphs, 1):
        # 将段落拆分成句子
        sentences = split_markdown(paragraph['content'])
        for sentence in sentences:
            # 构建字典
            result_dict = {
                'text': sentence,
                '段落': paragraph['content'],
                '文件名': file_name,
                '论文名':title,
                'title_level1': paragraph['title_level1'] if 'title_level1' in paragraph else None,
                'title_level2': paragraph['title_level2'] if 'title_level2' in paragraph else None,
                'title_level3': paragraph['title_level3'] if 'title_level3' in paragraph else None,
                'num_tokens':num_tokens,
                'serial_number':idx
            }

            # 添加到结果列表
            result_list.append(result_dict)
            # 添加文件名提问
            result_dict = {
                'text': '文献：'+file_name+' '+sentence,
                '段落': paragraph['content'],
                '文件名': file_name,
                '论文名':title,
                'title_level1': paragraph['title_level1'] if 'title_level1' in paragraph else None,
                'title_level2': paragraph['title_level2'] if 'title_level2' in paragraph else None,
                'title_level3': paragraph['title_level3'] if 'title_level3' in paragraph else None,
                'num_tokens':num_tokens,
                'serial_number':idx
            }
            result_list.append(result_dict)
    # 添加文件名，所有段落
    result_dict = {
        'text': file_name,
        '段落': paragraph['content'],
        '文件名': file_name,
        '论文名':title,
        'num_tokens':num_tokens
    }
    result_list.append(result_dict)

    return result_list


def extract_pdf(filepath,upload_url=None):
    upload_url = upload_url if upload_url else os.getenv("OCR_API_URL")
    filename = os.path.basename(filepath)
    files = {'file': (filename, open(filepath, 'rb'))}
    result = {'file_extension':'pdf', 'file_name':filename}
    response = requests.post(upload_url, data={'return_format': 'md'}, files=files)
    try:
        assert response.status_code == 200
        response_json = response.json()
        result['file_content'] = "".join(response_json['result'])
        return result
    except:
        print("status_code:", response.status_code)
        print(response.json())
        return []
    

def read_docx(filepath):
    """
    从.docx文件中读取文本内容。

    参数:
    file_path (str): .docx文件的路径。

    返回:
    str: 文件中的文本内容。
    """
    try:
        filename = os.path.basename(filepath)
        result = {'file_extension':'doc', 'file_name':filename}
        doc = docx.Document(filepath)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        full_text = [item for item in full_text if item != '']
        result['file_content'] = '\n'.join(full_text)
        return result
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ''

def process_docx(markdown_text, file_name):
    # 调用函数将Markdown拆分成段落
    paragraphs = split_docs_by_headings(markdown_text)
    print('file_name',file_name)
    base_name, _ = os.path.splitext(file_name)
    print('base_name',base_name)
    title = paragraphs[0]['content'] if len(paragraphs)>0 and 'content' in paragraphs[0] else base_name
    num_tokens = num_tokens_from_string(markdown_text, "cl100k_base")

    # 构建结果列表
    result_list = []
    for idx, paragraph in enumerate(paragraphs, 1):
        # 将段落拆分成句子
        sentences = split_paragraph_into_sentences(paragraph['content'])
        for sentence in sentences:
            # 构建字典
            result_dict = {
                'text': sentence,
                '段落': paragraph['content'],
                '文件名': file_name,
                '论文名':title,
                'title_level1': paragraph['title_level1'] if 'title_level1' in paragraph else None,
                'title_level2': paragraph['title_level2'] if 'title_level2' in paragraph else None,
                'title_level3': paragraph['title_level3'] if 'title_level3' in paragraph else None,
                'num_tokens':num_tokens,
                'serial_number':idx
            }
            

            # 添加到结果列表
            result_list.append(result_dict)
            # 添加文件名提问
            result_dict = {
                'text': '文献：'+file_name+' '+sentence,
                '段落': paragraph['content'],
                '文件名': file_name,
                '论文名':title,
                'title_level1': paragraph['title_level1'] if 'title_level1' in paragraph else None,
                'title_level2': paragraph['title_level2'] if 'title_level2' in paragraph else None,
                'title_level3': paragraph['title_level3'] if 'title_level3' in paragraph else None,
                'num_tokens':num_tokens,
                'serial_number':idx
            }
            result_list.append(result_dict)
    # 添加文件名，所有段落
    result_dict = {
        'text': file_name,
        '段落': paragraph['content'],
        '文件名': file_name,
        '论文名':title,
        'num_tokens':num_tokens
    }
    result_list.append(result_dict)

    return result_list

def read_txt(filepath):
    """
    从.txt文件中读取文本内容。

    参数:
    filepath (str): .txt文件的路径。

    返回:
    dict: 包含文件扩展名、文件名和文件内容的字典。
    """
    try:
        filename = os.path.basename(filepath)
        result = {'file_extension': 'txt', 'file_name': filename}
        with open(filepath, 'r', encoding='utf-8') as file:
            file_content = file.read()
        result['file_content'] = file_content
        return result
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ''

def process_and_deduplicate(data_list):
    # 去掉每个字典中的'text'字段
    processed_list = [{key: value for key, value in d.items() if key != 'text'} for d in data_list]

    # 对处理后的列表进行字典去重，确保'serial_number'不重复
    unique_list = [dict(t) for t in {tuple(d.items()) for d in processed_list}]

    # 检查是否包含'serial_number'键，然后按'serial_number'从小到大排序
    sorted_list = sorted(unique_list, key=lambda x: x.get('serial_number', 0))
    return sorted_list