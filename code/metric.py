# -*- coding: UTF-8 -*-
"""
@File    ：metric.py
@Author  ：zfk
@Date    ：2024/5/9 16:15
"""
import json
import openai
import nltk
from rouge import Rouge
from tenacity import retry, stop_after_attempt, wait_fixed

# 设置你的OpenAI API密钥
api_key = '————'

def metric(pred: str, answer_list: list[str]):
    # 计算BLEU
    print(pred, answer_list)
    reference = [answer.split() for answer in answer_list]
    candidate = pred.split()
    bleu_score = nltk.translate.bleu_score.sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

    # 计算ROUGE
    rouge = Rouge()
    rouge_scores = rouge.get_scores(pred, ' '.join(answer_list))

    return bleu_score, rouge_scores[0]['rouge-l']

@retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
def get_gpt4_scores(context, query, provided_answer, expected_answer):
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一个评估文本相关性和语义相似性的专家。"},
            {"role": "user", "content": f"上下文：\n{context}\n\n查询：\n{query}\n\n提供的答案：\n{provided_answer}\n\n期望的答案：\n{expected_answer}\n\n请根据以下评分标准进行评估，并按以下格式输出：\n\n1. 上下文与查询的相关性（0-10分）：\n- 10分：上下文完全相关，直接回答了查询中的问题，并提供了充足的信息。\n- 8-9分：上下文高度相关，包含大部分必要的信息，但可能缺少一些细节。\n- 6-7分：上下文较为相关，但包含的部分信息可能与查询无关，或缺少关键细节。\n- 4-5分：上下文有一定相关性，但包含大量无关信息，或缺少很多关键细节。\n- 2-3分：上下文很少相关，几乎没有提供有用的信息。\n- 0-1分：上下文完全不相关，没有提供任何有用的信息。\n\n2. 提供的答案与期望答案的语义等价性（0-10分）：\n- 10分：提供的答案与期望答案完全等价，无论是内容还是表达方式上都非常接近。\n- 8-9分：提供的答案与期望答案高度等价，内容基本一致，但可能有些微的表达差异。\n- 6-7分：提供的答案与期望答案较为等价，主要内容相同，但有一些次要的内容差异。\n- 4-5分：提供的答案与期望答案有一定等价性，但主要内容有较大的差异。\n- 2-3分：提供的答案与期望答案几乎没有等价性，主要内容差异很大。\n- 0-1分：提供的答案与期望答案完全不等价，内容完全不同。\n\n请按以下格式输出：\nContext Score: [0-10]\nSemantic Score: [0-10]\n"}
        ],
        max_tokens=256,
        temperature=0.5
    )

    gpt4_response = response.choices[0].message.content.strip()
    
    # 提取两个评分
    context_score = extract_score(gpt4_response, "Context Score")
    semantic_score = extract_score(gpt4_response, "Semantic Score")
    
    return context_score, semantic_score

def extract_score(response_text, score_type):
    lines = response_text.split("\n")
    for line in lines:
        if score_type in line:
            score = line.split(":")[-1].strip()
            try:
                return float(score)
            except ValueError:
                return 0.0
    return None

if __name__ == '__main__':
    start_line = 0
    with open('../data/test_gpt-3.5-turbo_bm25.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    bleu_scores = []
    rouge_l_p_scores = []
    rouge_l_r_scores = []
    rouge_l_f_scores = []
    context_scores = []
    semantic_scores = []

    output_file_path = '../data/score/test_gpt-3.5-turbo_bm25.jsonl'
    with open(output_file_path, 'a', encoding='utf-8') as f:
        for index, line in enumerate(lines[start_line:], start=start_line):
            data = json.loads(line)
            context = data.get('context', '')
            query = data.get('query', '')
            provided_answer = data.get('pred', '')
            expected_answer = data.get('answer', '')
            
            bleu, rouge_l = metric(provided_answer.lower().strip(), [expected_answer.lower().strip()])
            context_score, semantic_score = get_gpt4_scores(context, query, provided_answer, expected_answer)

            bleu_score = round(bleu, 4)
            rouge_l_p = round(rouge_l['p'], 4)
            rouge_l_r = round(rouge_l['r'], 4)
            rouge_l_f = round(rouge_l['f'], 4)

            # 将分数添加到列表中以计算平均值
            bleu_scores.append(bleu_score)
            rouge_l_p_scores.append(rouge_l_p)
            rouge_l_r_scores.append(rouge_l_r)
            rouge_l_f_scores.append(rouge_l_f)
            context_scores.append(context_score)
            semantic_scores.append(semantic_score)

            # 将分数放在前面，并重新组合行内容
            new_line = f"BLEU: {bleu_score} ROUGE-L P: {rouge_l_p} ROUGE-L R: {rouge_l_r} ROUGE-L F: {rouge_l_f} Context Score: {context_score} Semantic Score: {semantic_score} {json.dumps(data)}"
            
            # 写入文件并打印每一行的结果
            f.write(new_line + '\n')
            print(new_line)

            # 记录写入的行数
            print(f"已写入行数: {index + 1}")

    # 计算平均分
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge_l_p = sum(rouge_l_p_scores) / len(rouge_l_p_scores)
    avg_rouge_l_r = sum(rouge_l_r_scores) / len(rouge_l_r_scores)
    avg_rouge_l_f = sum(rouge_l_f_scores) / len(rouge_l_f_scores)
    avg_context_score = sum(context_scores) / len(context_scores)
    avg_semantic_score = sum(semantic_scores) / len(semantic_scores)

    # 打印平均分
    avg_scores_line = f"Average BLEU: {avg_bleu} Average ROUGE-L P: {avg_rouge_l_p} Average ROUGE-L R: {avg_rouge_l_r} Average ROUGE-L F: {avg_rouge_l_f} Average Context Score: {avg_context_score} Average Semantic Score: {avg_semantic_score}"
    print(avg_scores_line)

    # 将平均分写入文件
    with open(output_file_path, 'a', encoding='utf-8') as f:
        f.write(avg_scores_line + '\n')

    print("评分结果已写入文件。")
