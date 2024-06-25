import os
import json

def is_average_line(line):
    return "Average BLEU" in line

def read_and_calculate_average(file_path):
    bleu_scores = []
    rouge_l_p_scores = []
    rouge_l_r_scores = []
    rouge_l_f_scores = []
    context_scores = []
    semantic_scores = []

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 检查最后一行是否为平均值
    last_line = lines[-1].strip()
    if is_average_line(last_line):
        print(f"{file_path}: 文件末尾已有平均值数据，不会重复写入。")
        return

    for line in lines:
        try:
            # 提取有效的JSON部分进行解析
            json_part = line.split(' {', 1)[-1]
            data = json.loads('{' + json_part)
            
            # 提取分数
            bleu_score = float(line.split('BLEU: ')[1].split(' ')[0])
            rouge_l_p = float(line.split('ROUGE-L P: ')[1].split(' ')[0])
            rouge_l_r = float(line.split('ROUGE-L R: ')[1].split(' ')[0])
            rouge_l_f = float(line.split('ROUGE-L F: ')[1].split(' ')[0])
            context_score = float(line.split('Context Score: ')[1].split(' ')[0])
            semantic_score = float(line.split('Semantic Score: ')[1].split(' ')[0])

            # 将None值设为0.0
            bleu_scores.append(bleu_score if bleu_score is not None else 0.0)
            rouge_l_p_scores.append(rouge_l_p if rouge_l_p is not None else 0.0)
            rouge_l_r_scores.append(rouge_l_r if rouge_l_r is not None else 0.0)
            rouge_l_f_scores.append(rouge_l_f if rouge_l_f is not None else 0.0)
            context_scores.append(context_score if context_score is not None else 0.0)
            semantic_scores.append(semantic_score if semantic_score is not None else 0.0)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
        except (ValueError, IndexError) as e:
            print(f"解析分数错误: {e}")

    # 计算平均值
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_rouge_l_p = sum(rouge_l_p_scores) / len(rouge_l_p_scores) if rouge_l_p_scores else 0.0
    avg_rouge_l_r = sum(rouge_l_r_scores) / len(rouge_l_r_scores) if rouge_l_r_scores else 0.0
    avg_rouge_l_f = sum(rouge_l_f_scores) / len(rouge_l_f_scores) if rouge_l_f_scores else 0.0
    avg_context_score = sum(context_scores) / len(context_scores) if context_scores else 0.0
    avg_semantic_score = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0.0

    avg_scores = {
        "Average BLEU": avg_bleu,
        "Average ROUGE-L P": avg_rouge_l_p,
        "Average ROUGE-L R": avg_rouge_l_r,
        "Average ROUGE-L F": avg_rouge_l_f,
        "Average Context Score": avg_context_score,
        "Average Semantic Score": avg_semantic_score
    }

    # 将平均值写入文件
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(avg_scores) + '\n')

    print(f"{file_path}: 平均值已写入文件。")

def process_all_jsonl_files():
    current_directory = os.getcwd()
    for filename in os.listdir(current_directory):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(current_directory, filename)
            read_and_calculate_average(file_path)

# 运行脚本
process_all_jsonl_files()
