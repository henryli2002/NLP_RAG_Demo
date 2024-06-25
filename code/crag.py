# -*- coding: UTF-8 -*-
"""
@File    ：crag.py
@Author  ：zfk
@Date    ：2024/5/9 12:55
"""
import json
import os
import time
from llama_index.core import Document, Settings
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 引入 OpenaiApiLLM 和 MyApiLLM
from model_response import MyApiLLM, OpenaiApiLLM

chunk_size = 100
start_line = 0
max_retries = 10
retry_wait_time = 10  # seconds

def build_automerging_index(documents, save_dir="merging_index", chunk_sizes=None):
    if not os.path.exists(save_dir):
        print('creating index directory', save_dir)
        chunk_sizes = chunk_sizes or [2048, 512, 128]
        node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
        nodes = node_parser.get_nodes_from_documents(documents)
        nodes = [node for node in nodes if node.text.strip()]
        leaf_nodes = get_leaf_nodes(nodes)
        leaf_nodes = [leaf for leaf in leaf_nodes if leaf.text.strip()]
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)
        automerging_index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        print('loading index directory', save_dir)
        automerging_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=save_dir))
    return automerging_index

def process_chunk(chunk, chunk_index, result_dir, similarity_top_k=12):
    result = []
    total_time = 0
    count = 0
    for i, line in enumerate(chunk):
        data = json.loads(line)

        query = data['query']
        answer = data['answer']
        search_results = data['search_results']
        documents = []
        total_characters = 0
        for search_result in search_results:
            if 'page_result' in search_result:
                text = search_result['page_result']
                total_characters += len(text)
                documents.append(Document(text=text))
        retry_count = 0
        while retry_count < max_retries:
            try:
                start_time = time.time()
                index = build_automerging_index(documents, save_dir=f'merging_index/{chunk_index*chunk_size+i}.index')
                base_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
                retriever = AutoMergingRetriever(base_retriever, index.storage_context, verbose=True)
                end_time = time.time()
                total_time += (end_time - start_time)
                count+=1
                auto_merging_engine = RetrieverQueryEngine.from_args(retriever, node_postprocessors=[Settings.rerank_model])
                auto_merging_response, prompt = auto_merging_engine.query(query)
                if not auto_merging_response or not auto_merging_response.response:
                    raise ValueError("Empty response")
                print(f'{query=}')
                print(f'{prompt=}')
                print(f'{auto_merging_response.response=}')
                print(f'{answer=}')
                data['pred'] = auto_merging_response.response
                result.append(json.dumps({'prompt': prompt, 'answer': answer, 'pred': auto_merging_response.response}, ensure_ascii=False) + '\n')
                break
            except Exception as e:
                print(f"Error processing line {i} in chunk {chunk_index}: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Retrying ({retry_count}/{max_retries})...")
                    time.sleep(retry_wait_time)
                else:
                    print(f"Failed after {max_retries} retries. Skipping...")
    if count > 0:
        average_time = total_time / count
        print(f"平均处理时间: {average_time:.4f} 秒")
    else:
        print("没有处理任何数据")
    result_file = os.path.join(result_dir, f'result_chunk_{chunk_index}.jsonl')
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(''.join(result))

def test_crag(source_file='../data/test_rag.jsonl', result_dir='../data/results', start_line=0):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i in range(start_line, len(lines), chunk_size):
        chunk = lines[i:i + chunk_size]
        process_chunk(chunk, i // chunk_size, result_dir)

def merge_results(result_dir='../data/results', target_file='../data/test_rag_gpt-3.5-turbo_finetune.jsonl'):
    result_files = [os.path.join(result_dir, f) for f in os.listdir(result_dir) if f.startswith('result_chunk_')]
    merged_results = []
    for result_file in result_files:
        with open(result_file, 'r', encoding='utf-8') as f:
            merged_results.extend(f.readlines())
    
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(''.join(merged_results))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='api', choices=['api', 'openai'])
    parser.add_argument('--api_key', type=str, help='API key for custom API', default='')
    parser.add_argument('--secret_key', type=str, help='Secret key for custom API', default='')
    parser.add_argument('--openai_api_key', type=str, help='OpenAI API key', default='')
    parser.add_argument('--embedding_model_path', type=str, help='Local embedding model path', default='../BAAI/bge-small-en-v1.5')
    parser.add_argument('--similarity_top_k', type=int, default=12)
    parser.add_argument('--data_path', type=str, help='Local data path', default='../data/Elon.txt')
    parser.add_argument('--save_path', type=str, help='Chunk save path', default='./merging_index')
    parser.add_argument('--rerank_model_path', type=str, help='Local rerank model path', default='../BAAI/bge-reranker-base')
    parser.add_argument('--rerank_top_n', type=int, default=2)
    args = parser.parse_args()

    # 根据模型类型选择 LLM
    if args.model_type == 'openai':
        assert args.openai_api_key, "OpenAI API key must be provided"
        llm = OpenaiApiLLM(args.openai_api_key)
    else:
        assert args.api_key and args.secret_key, "API key and Secret key must be provided"
        llm = MyApiLLM(args.api_key, args.secret_key)

    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name=args.embedding_model_path)
    Settings.rerank_model = SentenceTransformerRerank(top_n=args.rerank_top_n, model=args.rerank_model_path)
    test_crag(start_line=start_line)
    merge_results()
