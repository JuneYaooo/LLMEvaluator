import os
import dotenv
import datetime
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.run_config import RunConfig
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
)

from src.ragas.common.read_files import process_markdown, extract_pdf, read_docx, process_docx, process_and_deduplicate, read_txt
from src.ragas.testset.generator import TestsetGenerator
from src.ragas.testset.evolutions import (
    simple,
    reasoning,
    counterfactual,
    error_correction,
    MultiContextEvolution,
    KContextEvolution
)

env_state = os.getenv('ENV_STATE', 'example')
dotenv.load_dotenv(f'.env.{env_state}', override=True)


def get_file_info(file_path):
    file_extension = file_path.split('.')[-1].lower()
    file_info = []

    if file_extension == 'pdf':
        file_info = extract_pdf(file_path, upload_url=os.getenv('OCR_API_URL'))
        if not file_info:
            raise ValueError(f"Failed to read file {file_path}")
        result_list = process_markdown(file_info["file_content"], file_info["file_name"])
        sorted_list = process_and_deduplicate(result_list)

    elif file_extension == 'docx':
        file_info = read_docx(file_path)
        if not file_info:
            raise ValueError(f"Failed to read file {file_path}")
        result_list = process_docx(file_info["file_content"], file_info["file_name"])
        sorted_list = process_and_deduplicate(result_list)

    elif file_extension == 'txt':
        file_info = read_txt(file_path)
        if not file_info:
            raise ValueError(f"Failed to read file {file_path}")

    return file_info

def load_documents():
    documents = []
    for root, dirs, files in os.walk(os.getenv('DOC_DIR')):
        for file in files:
            if file.endswith('.pdf') or file.endswith('.docx') or file.endswith('.txt'):
                file_path = os.path.join(root, file)
                file_info = get_file_info(file_path)
                document = Document(page_content=file_info['file_content'], metadata={"filename": file_info["file_name"]}) #这里必须要filename，后面document里是根据这个名字来获取的
                documents.append(document)
    return documents

def generate_qa():
    documents = load_documents()
    k_context = KContextEvolution(context_num=2)
    multi_context = MultiContextEvolution(context_num=2)
    distributions = {multi_context: 1} #{simple: 0.5, reasoning: 0.25, multi_context: 0.25, counterfactual:0.7,error_correction}

    generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
    critic_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
    embeddings = OpenAIEmbeddings()
    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        embeddings,
        chunk_size=int(os.getenv('CHUNK_MAX_LENGTH'))
    )
    testset = generator.generate_with_langchain_docs(
        documents,
        test_size=5,
        raise_exceptions=False,
        with_debugging_logs=False,
        distributions=distributions,
        run_config=RunConfig(max_workers=1)
    )


    test_res_df = testset.to_pandas()
    
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists('output'):
        os.makedirs('output')
    test_res_df.to_csv(f"output/test_data_{current_time}.csv", index=False, encoding='utf-8')

def evaluate_qa():
    amnesty_qa = load_dataset("/mnt/workspace/maokangkun/RAG_Eval/amnesty_qa", "english_v2", trust_remote_code=True)
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
    embeddings = OpenAIEmbeddings()

    result = evaluate(
        amnesty_qa["eval"],
        metrics=[
            faithfulness,
            answer_relevancy
        ],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=1)
    )
    df = result.to_pandas()
    # print(df.head())

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"output/eval_{current_time}.csv", index=False, encoding='utf-8')

# generate_qa()
evaluate_qa()
