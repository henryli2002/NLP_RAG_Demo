import json

def load_queries_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        queries = {json.loads(line)['query'] for line in lines}
    return queries

def extract_matching_lines(source_file, test_file, output_file):
    # Load queries from the test file
    test_queries = load_queries_from_file(test_file)

    # Open the source file and output file
    with open(source_file, 'r', encoding='utf-8') as src_file, open(output_file, 'w', encoding='utf-8') as out_file:
        for line in src_file:
            data = json.loads(line)
            query = data['query']
            if query in test_queries:
                out_file.write(line)

if __name__ == '__main__':
    source_file = '../data/crag_data_2735.jsonl'  # 你的源文件
    test_file = '../data/test.jsonl'      # 包含需要匹配的queries的测试文件
    output_file = '../data/test_rag.jsonl'  # 输出文件

    extract_matching_lines(source_file, test_file, output_file)