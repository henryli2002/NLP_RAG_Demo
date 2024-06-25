import os
import openai
import json
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
from rank_bm25 import BM25Okapi



few_shot=0
rag = 1
method = 'tf-idf'

def split_text_to_sentences(text):
    import re
    # 使用正则表达式按照标点符号和换行符切割文本
    sentences = re.split(r'(?<=[.!?]) +|\n\n', text)
    # 去除空白句子并返回
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def get_top_k_similar_sentences(text, query_sentence, k=12, method=method):
    # 将文本按句子切割
    sentences = split_text_to_sentences(text)
    # 将查询句子和切割后的句子列表组合在一起
    all_sentences = [query_sentence] + sentences

    if method == 'tf-idf':
        # 使用TF-IDF向量化句子
        vectorizer = TfidfVectorizer().fit_transform(all_sentences)
        vectors = vectorizer.toarray()
        # 计算查询句子与所有其他句子的余弦相似度
        cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    elif method == 'bm25':
        # 使用BM25向量化句子
        tokenized_corpus = [doc.split(" ") for doc in all_sentences]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = query_sentence.split(" ")
        # 计算查询句子与所有其他句子的BM25分数
        bm25_scores = bm25.get_scores(query_tokens)[1:]  # Skip the query sentence itself
        cosine_similarities = np.array(bm25_scores)
    else:
        raise ValueError("Unsupported method. Choose 'tfidf' or 'bm25'.")

    # 获取前k个相似句子的索引
    top_k_indices = cosine_similarities.argsort()[-k:][::-1]
    # 提取前k个相似句子
    prompt = query_sentence + "\n\nContext:\n"
    for i in top_k_indices:
        prompt += sentences[i] + "\n"

    return prompt




def call_openai_api(queries, api_key, model, max_tokens, system_message):
    client = openai.OpenAI(api_key=api_key)
    responses = []
    examples = [
    {"query": "which player has the most career assists in the nba among players who have never been named to an all-star game?", "answer": "andre miller has the most career assists in the nba among players who have never been named to an all-star game, with 8,524 assists."},
    {"query": "what's the name of the movie that received the oscar for the best documentary feature film in 1995?", "answer": "maya lin: a strong clear vision"},
    {"query": "which movie won the academy award for best picture in 2018, categorized under the fantasy genre?", "answer": "the shape of water"},
    {"query": "how many animated movies has reese witherspoon been in?", "answer": "reese witherspoon has been in 4 animated movies."},
    {"query": "when was the oldest company in the dow jones added to the index?", "answer": "the oldest company in the dow jones, procter & gamble, was added on may 6, 1932."}
]
    for idx, query in enumerate(queries):
        if few_shot==1:
            messages = [
            {"role": "system", "content": system_message},]
            for example in examples:
                messages.append({"role": "user", "content": example['query']})
                messages.append({"role": "assistant", "content": example['answer']})
            messages.append({"role": "user", "content": query})
        else:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.5
        )
        answer = response.choices[0].message.content.strip()
        responses.append(answer)
        # 打印进度
        print(f"Query {idx + 1}/{len(queries)}: {query}")
        print(f"Response: {answer}\n")
    return responses

def process_file(input_file, api_key, model, max_tokens, batch_size, system_message):
    output_file = f'..\\data\\test_{model}_{method}.jsonl'
    queries = []
    original_data = []
    total_time = 0
    count = 0
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line)
            if rag == 1:
                query = data['query']
                text = " ".join([result['page_snippet'] for result in data['search_results']])
                start_time = time.time()
                prompt = get_top_k_similar_sentences(text, query)
                end_time = time.time()
                queries.append(prompt)
                data['prompt'] = prompt
                total_time += (end_time - start_time)
                count+=1
            else:
                queries.append(data['query'])
            original_data.append(data)
    if count > 0:
        average_time = total_time / count
        print(f"平均处理时间: {average_time:.4f} 秒")
    else:
        print("没有处理任何数据")
    results = []
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i+batch_size]
        batch_responses = call_openai_api(batch_queries, api_key, model, max_tokens, system_message)
        for j in range(len(batch_queries)):
            if rag==1:
                results.append({
                    'prompt': original_data[i + j]['prompt'],
                    'answer': original_data[i + j]['answer'],
                    'pred': batch_responses[j]
                })
            else:
                results.append({
                    'query': original_data[i + j]['query'],
                    'answer': original_data[i + j]['answer'],
                    'pred': batch_responses[j]
                })

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for result in results:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Process jsonl files with OpenAI API.')
    parser.add_argument('--input', required=True, help='Input jsonl file path')
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--model', required=True, help='OpenAI model to use')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum tokens per API call')
    parser.add_argument('--batch_size', type=int, default=5, help='Number of queries per batch')
    parser.add_argument('--system_message', required=True, help='System message prompt for the assistant')

    args = parser.parse_args()

    # 移除API密钥中的任何空格或换行符
    api_key = args.api_key.strip()

    process_file(args.input, api_key, args.model, args.max_tokens, args.batch_size, args.system_message)

if __name__ == '__main__':
    main()
