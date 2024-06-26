import os
import re
import tqdm
import json
import queue
import dotenv
import datasets
import datetime
import requests
import pandas as pd
from rich import pretty
from rich import inspect
from rich import print as rp
from pathlib import Path
from retrying import retry
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.run_config import RunConfig
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    answer_similarity
)
from ragas.testset.prompts import question_answer_prompt

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

pretty.install()
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
    k_context = KContextEvolution(context_num=3)
    multi_context = MultiContextEvolution(context_num=3)
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
        run_config=RunConfig(max_workers=int(os.getenv('MAX_WORKER')))
    )

    test_res_df = testset.to_pandas()

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists('output'):
        os.makedirs('output')
    test_res_df.to_csv(f"output/test_data_{current_time}.csv", index=False, encoding='utf-8')

def answer_qa(i=0):
    import sys
    sys.path.append('/ailab/user/maokangkun/Model_Eval/LLaMA-Factory')
    from src.llmtuner import ChatModel

    test_qa_file = sorted(Path('data/llm_questions').glob('*.xlsx'))[-1]
    test_qa = pd.read_excel(test_qa_file, sheet_name=1)
    print(len(test_qa))

    model_list = ['Meta-Llama-3-8B-Instruct', 'Qwen1.5-7B-Chat', 'pulse_v12_7b_gpt4_hf']
    model_name = model_list[i]
    model_path = f'/ailab/user/maokangkun/models/hf/{model_name}'
    device = f'cuda:{i}'
    args = {
        'model_name_or_path': model_path,
        'template': 'default',
        'flash_attn': 'fa2',
        'max_new_tokens': 256,
        'export_device': device,
        # 'export_dir': '/tmp',
    }
    if 'pulse' in model_name: del args['flash_attn']
    chat_model = ChatModel(args)

    data = []
    for _, d in tqdm.tqdm(list(test_qa.iterrows())):
        contexts = [c for c in eval(d.contexts) if c]
        context = '\n'.join([f'[context {i+1}]. {c}' for i, c in enumerate(contexts)])
        if context:
            context = '\n'+context
        else:
            context = 'None'

        inp = question_answer_prompt.format(
            question=d.question,
            context=context
        ).prompt_str

        messages = [{"role": "user", "content": inp}]

        retry = 0
        while retry <= 5:
            ret = chat_model.chat(messages)[0].response_text
            m = re.findall(r'```{"answer": "(.*)", "verdict": ".*"}```', ret)
            if m:
                ans = m[0]
                break
            else:
                ans = ret
                retry += 1

        data.append({
            'id': d.order_number,
            'question': d.question,
            'answer': ans,
            'contexts': contexts,
            'ground_truth': d.ground_truth,
            'metadata': [i for i in eval(d.metadata) if i],
            'evolution_type': d.evolution_type,
        })

    out_file = f'data/llm_answers/{test_qa_file.stem}_{model_name}_answer.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def evaluate_qa():
    # model_name = "gpt-3.5-turbo-16k"
    # model_name = "Qwen1.5-7B-Chat"
    # model_name = "Meta-Llama-3-8B-Instruct"
    model_name = "Meta-Llama-70B-Instruct"

    answer_dir = Path('data/llm_answers')
    eval_dir = Path('data/llm_evals')
    qa_id = sorted(Path('data/llm_questions').glob('*.xlsx'))[-1].stem
    tobe_eval = []
    for ans_file in answer_dir.glob(f'{qa_id}_*.json'):
        eval_file = eval_dir / (ans_file.stem.replace('_answer', '') + f'_by_{model_name}.csv')
        if not eval_file.exists() and model_name not in ans_file.name:
            tobe_eval.append([ans_file, eval_file])

    for inp_file, out_file in tobe_eval:
        print(inp_file)
        eval_qa = datasets.load_dataset("json", data_files={'eval':str(inp_file)})

        llm = ChatOpenAI(model=model_name)
        embeddings = OpenAIEmbeddings()

        result = evaluate(
            eval_qa["eval"],
            metrics=[
                faithfulness,
                answer_relevancy,
                answer_similarity
            ],
            llm=llm,
            embeddings=embeddings,
            run_config=RunConfig(
                max_workers=int(os.getenv('MAX_WORKER')),
                max_wait=1200,
                timeout=1200
            ),
            raise_exceptions=False
        )
        df = result.to_pandas()
        # print(df.head())
        df.to_csv(out_file, index=False, encoding='utf-8')

@retry(stop_max_attempt_number=10, wait_random_min=200, wait_random_max=400)
def get_pulse_reply(prompt, model='base_model'):
    api_url = 'https://mchatgpt-internal.dev.6ccloud.com/v1/api/completion/generate_stream'
    headers = {
        'content-type': 'application/json',
        'authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIzZjM0ODE2YS0xYTFmLTQ2ZTMtYWM5ZS00N2IzYWI5ZWI4ZjkiLCJleHAiOjIwMDQxOTQ1ODcsInNjb3BlcyI6WyJ1c2VyIl19.aBHWyQOt-yF-mIkG0RTm1i2j-46ZDQ7y7kdEqSBI2v8'
    }
    json_data = {
        'action': 'To user',
        'parent_messages': [
            {
                'action': 'From user',
                'content': prompt
            }
        ],
        'gen_kwargs': {
            'model': model,
            'num_return_sequences': 1,
            'max_new_tokens': 256,
        },
    }

    try:
        with requests.post(api_url, headers=headers, json=json_data, stream=True, timeout=60) as res:
            for data in res.iter_lines(decode_unicode=True):
                if data and data != 'data: [DONE]': ret = data
            return json.loads(ret[6:])['messages'][0]['content']['parts'][0]
    except:
        return None

def answer_qa_pulse():
    test_qa_file = sorted(Path('data/llm_questions').glob('*.xlsx'))[-1]
    test_qa = pd.read_excel(test_qa_file, sheet_name=1)
    print(len(test_qa))

    model_name = '123bv11'

    data = []
    for _, d in tqdm.tqdm(list(test_qa.iterrows())):
        contexts = [c for c in eval(d.contexts) if c]
        context = '\n'.join([f'[context {i+1}]. {c}' for i, c in enumerate(contexts)])
        if context:
            context = '\n'+context
        else:
            context = 'None'

        inp = question_answer_prompt.format(
            question=d.question,
            context=context
        ).prompt_str

        retry = 0
        while retry <= 5:
            ret = get_pulse_reply(inp, model_name)
            m = re.findall(r'```{"answer": "(.*)", "verdict": ".*"}```', ret)
            if m:
                ans = m[0]
                break
            else:
                ans = ret
                retry += 1

        data.append({
            'id': d.order_number,
            'question': d.question,
            'answer': ans,
            'contexts': contexts,
            'ground_truth': d.ground_truth,
            'metadata': [i for i in eval(d.metadata) if i],
            'evolution_type': d.evolution_type,
        })
    out_file = f'data/llm_answers/{test_qa_file.stem}_{model_name}_answer.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def plot_result():
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_theme(style="whitegrid")
    eval_dir = Path('data/llm_evals')

    qv = sorted(Path('data/llm_questions').glob('*.xlsx'))[-1].stem
    data = pd.DataFrame([], columns=['question_id', 'model', 'evolution_type', 'metric', 'value'])
    for eval_file in eval_dir.glob(f'{qv}_*.csv'):
        df = pd.read_csv(eval_file)
        model = eval_file.stem.split('_by_')[0][len(qv)+1:]
        for _,row in df.iterrows():
            data.loc[len(data)] = [row.id, model, row.evolution_type, 'faithfulness', row.faithfulness]
            data.loc[len(data)] = [row.id, model, row.evolution_type, 'answer_relevancy', row.answer_relevancy]
            data.loc[len(data)] = [row.id, model, row.evolution_type, 'answer_similarity', row.answer_similarity]

    g = sns.catplot(
        data=data, kind="bar",
        x="metric", y="value", hue="model",
        palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("", "")

    plt.savefig(eval_dir / f'{qv}_overall.png', dpi=300)

    evo_type = list(set(data.evolution_type))
    _, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        d = data[data.evolution_type==evo_type[i]]
        sns.barplot(x="metric", y="value", data=d, hue="model", ax=ax)
        ax.get_legend().remove()
        ax.set_title(evo_type[i])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='x', labelrotation=-45)

    plt.legend(loc="upper left", fontsize="8", bbox_to_anchor=(1.05, 1.05))
    plt.savefig(eval_dir / f'{qv}_items.png', dpi=300)

# generate_qa()
# answer_qa(1)
# answer_qa_pulse()
# evaluate_qa()
plot_result()
