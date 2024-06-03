import os
import openai
import json
import argparse

def call_openai_api(queries, api_key, model, max_tokens, system_message):
    client = openai.OpenAI(api_key=api_key)
    responses = []
    for idx, query in enumerate(queries):
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
    output_file = f'..\\data\\test_{model}.jsonl'
    queries = []
    original_data = []

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line)
            queries.append(data['query'])
            original_data.append(data)

    results = []
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i+batch_size]
        batch_responses = call_openai_api(batch_queries, api_key, model, max_tokens, system_message)
        for j in range(len(batch_queries)):
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
