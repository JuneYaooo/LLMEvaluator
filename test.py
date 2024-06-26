from src.ragas.common.read_files import read_file, process_markdown, list_files_in_folder, unzip_file, extract_pdf, read_docx, process_docx, process_and_deduplicate,read_txt
import os
import dotenv
from langchain_core.documents import Document

from src.ragas.testset.generator import TestsetGenerator
from src.ragas.testset.evolutions import simple, reasoning, multi_context, counterfactual, error_correction, no_reference, negative_rejection, noise_robustness,MultiContextEvolution,KContextEvolution


dotenv.load_dotenv(f'.env', override=True)

print('os.getenv("LLM_API_BASE")',os.getenv("LLM_API_BASE"))
os.environ['OPENAI_API_BASE'] = os.getenv("LLM_API_BASE")
os.environ["OPENAI_API_KEY"] = os.getenv("LLM_API_KEY")

OCR_API_URL = os.getenv("OCR_API_URL")

def get_file_info(file_path):
    file_extension = file_path.split('.')[-1].lower()
    file_info = []

    if file_extension == 'pdf':
        file_info = extract_pdf(file_path, upload_url=OCR_API_URL)
        if not file_info:
            raise ValueError(f"Failed to read file {file_path}")
    elif file_extension == 'docx':
        file_info = read_docx(file_path)
        if not file_info:
            raise ValueError(f"Failed to read file {file_path}")
    elif file_extension == 'txt':
        file_info = read_txt(file_path)
        if not file_info:
            raise ValueError(f"Failed to read file {file_path}")

    return file_info

# 要遍历的文件夹路径
folder_path = '/home/ubuntu/github/LLMEvaluator/data/test_data_pdf'#'/home/ubuntu/github/ragas/data' #/home/ubuntu/github/LLMEvaluator/data

# 存储所有文档的列表
documents = []

# 遍历文件夹中的所有文件
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.pdf') or file.endswith('.docx') or file.endswith('.txt'):
            file_path = os.path.join(root, file)
            file_info = get_file_info(file_path)
            document = Document(page_content=file_info['file_content'], metadata={"filename": file_info["file_name"]}) #这里必须要filename，后面document里是根据这个名字来获取的
            documents.append(document)

k_context = KContextEvolution(context_num=5)
# 多来源
multi_context = MultiContextEvolution(context_num=3)
# 这里选择要生成什么类型的，以及比例
distributions = {reasoning:1}#{no_reference:0.1,simple: 0.1, negative_rejection:0.1,k_context:0.2,noise_robustness:0.1,multi_context:0.1, reasoning: 0.1, counterfactual:0.1,error_correction:0.1 } 

generator = TestsetGenerator.with_openai(chunk_size=1000)
testset = generator.generate_with_langchain_docs(
    documents[:50],
    test_size=20,
    raise_exceptions=False,
    with_debugging_logs=False,
    distributions=distributions,
)


test_res_df = testset.to_pandas()
# print('test_res_df',test_res_df)
import datetime

# 获取当前时间
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# 检查output文件夹是否存在
if not os.path.exists('output'):
    # 如果不存在，创建output文件夹
    os.makedirs('output')
test_res_df.to_csv(f"output/test_data_{current_time}.csv", index=False, encoding='utf-8')
print('end')