import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer, scoring
from tqdm import tqdm
from inference import generate_caption_from_image  # 引入你已实现的图像生成描述函数

# 确保 punkt 资源包已经下载，用于分词
nltk.download('punkt')

# 创建 img 目录以保存图表
os.makedirs('imgs', exist_ok=True)


def load_ground_truths(file_path):
    ground_truths = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # 每行由图像文件名和描述组成，用空格隔开
                image_id, caption = line.split(' ', 1)

                # 如果图像ID还没有添加到字典中，则初始化一个列表
                if image_id not in ground_truths:
                    ground_truths[image_id] = []

                # 将描述添加到图像ID对应的描述列表中
                ground_truths[image_id].append(caption)

    return ground_truths


# 计算 BLEU-1 和 BLEU-4 分数
def calculate_bleu(references, hypothesis):
    smoothie = SmoothingFunction().method4
    reference_tokens = [nltk.word_tokenize(ref.lower()) for ref in references]
    hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())

    bleu1_score = sentence_bleu(reference_tokens, hypothesis_tokens, weights=[
                                1, 0, 0, 0], smoothing_function=smoothie)  # BLEU-1
    bleu4_score = sentence_bleu(reference_tokens, hypothesis_tokens, weights=[
                                0.25, 0.25, 0.25, 0.25], smoothing_function=smoothie)  # BLEU-4

    return bleu1_score, bleu4_score

# 计算 ROUGE 分数


def calculate_rouge(references, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    reference_str = ' '.join(references)  # 将参考描述连接成一个字符串
    hypothesis_str = hypothesis  # 生成的描述

    rouge_scores = scorer.score(reference_str, hypothesis_str)

    return {
        'rouge1': rouge_scores['rouge1'].fmeasure,  # 取F1分数
        'rougeL': rouge_scores['rougeL'].fmeasure   # 取F1分数
    }

# 对单张图片进行评估


def evaluate_image(image_path, ground_truth_captions):
    generated_caption = generate_caption_from_image(
        image_path)  # 调用 inference.py 中的函数

    # 计算 BLEU-1 和 BLEU-4 分数
    bleu1_score, bleu4_score = calculate_bleu(
        ground_truth_captions, generated_caption)

    # 计算 ROUGE 分数
    rouge_scores = calculate_rouge(ground_truth_captions, generated_caption)

    return bleu1_score, bleu4_score, rouge_scores

# 对整个数据集进行评估


def evaluate_dataset(image_ids, ground_truths, split_name):
    total_bleu1 = 0
    total_bleu4 = 0
    total_rouge = {'rouge1': 0, 'rougeL': 0}
    num_images = len(image_ids)

    for image_id in tqdm(image_ids, desc=f"Evaluating {split_name} set"):
        image_path = f"dataset/flickr8k/Images/{image_id}"
        ground_truth_captions = ground_truths.get(
            image_id, ["No ground truth available."])

        # 评估单张图片，返回 BLEU-1、BLEU-4 和 ROUGE 分数
        bleu1_score, bleu4_score, rouge_scores = evaluate_image(
            image_path, ground_truth_captions)

        total_bleu1 += bleu1_score
        total_bleu4 += bleu4_score
        for key in total_rouge:
            total_rouge[key] += rouge_scores[key]

    # 计算平均分数
    avg_bleu1 = total_bleu1 / num_images
    avg_bleu4 = total_bleu4 / num_images
    avg_rouge = {key: total_rouge[key] / num_images for key in total_rouge}

    print(f"\n{split_name} Set Evaluation:")
    print(f"Average BLEU-1 Score: {avg_bleu1}")
    print(f"Average BLEU-4 Score: {avg_bleu4}")
    print(f"Average ROUGE Scores: {avg_rouge}")

    return avg_bleu1, avg_bleu4, avg_rouge

# 生成并保存图表


def plot_scores(bleu1_scores, bleu4_scores, rouge1_scores, rougeL_scores):
    datasets = ['Train', 'Validation', 'Test']
    x = np.arange(len(datasets))

    # 创建 BLEU-1 和 BLEU-4 分数的柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.1, bleu1_scores, 0.2, label='BLEU-1', color='b', alpha=0.6)
    plt.bar(x + 0.1, bleu4_scores, 0.2, label='BLEU-4', color='g', alpha=0.6)

    plt.xticks(x, datasets)
    plt.ylabel('BLEU Scores')
    plt.title('BLEU-1 and BLEU-4 Score Comparison')
    plt.legend()
    plt.savefig('imgs/bleu_score_comparison.png')  # 保存 BLEU 分数对比图
    plt.close()

    # 创建 ROUGE 分数的柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.1, rouge1_scores, 0.2, label='ROUGE-1', color='r', alpha=0.6)
    plt.bar(x + 0.1, rougeL_scores, 0.2, label='ROUGE-L', color='b', alpha=0.6)

    plt.xticks(x, datasets)
    plt.ylabel('ROUGE Scores')
    plt.title('ROUGE-1 and ROUGE-L Score Comparison')
    plt.legend()
    plt.savefig('imgs/rouge_score_comparison.png')  # 保存 ROUGE 分数对比图
    plt.close()


# 生成并保存分数表格
def plot_score_table(bleu1_scores, bleu4_scores, rouge1_scores, rougeL_scores):
    datasets = ['Score', 'Train', 'Validation', 'Test']  # 第一列抬头为 'Score'

    # 创建表格数据，第一列为评分名称
    data = [
        ["BLEU-1", "%.3f" % bleu1_scores[0], "%.3f" %
            bleu1_scores[1], "%.3f" % bleu1_scores[2]],
        ["BLEU-4", "%.3f" % bleu4_scores[0], "%.3f" %
            bleu4_scores[1], "%.3f" % bleu4_scores[2]],
        ["ROUGE-1", "%.3f" % rouge1_scores[0], "%.3f" %
            rouge1_scores[1], "%.3f" % rouge1_scores[2]],
        ["ROUGE-L", "%.3f" % rougeL_scores[0], "%.3f" %
            rougeL_scores[1], "%.3f" % rougeL_scores[2]]
    ]

    # 创建表格
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')

    # 添加表格
    table = ax.table(
        cellText=data,
        colLabels=datasets,  # 包含 'Score' 列抬头
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)  # 调整比例

    # 保存表格图片
    plt.savefig('imgs/score_table.png')  # 保存表格图
    plt.close()


if __name__ == "__main__":
    # 创建命令行解析器
    parser = argparse.ArgumentParser(
        description="BLEU-1, BLEU-4 and ROUGE score evaluation or chart generation.")

    # 添加命令行参数，用于输入 BLEU 和 ROUGE 分数
    parser.add_argument("--bleu1", type=float, nargs=3,
                        help="Input BLEU-1 scores for Train, Validation, and Test sets.")
    parser.add_argument("--bleu4", type=float, nargs=3,
                        help="Input BLEU-4 scores for Train, Validation, and Test sets.")
    parser.add_argument("--rouge1", type=float, nargs=3,
                        help="Input ROUGE-1 scores for Train, Validation, and Test sets.")
    parser.add_argument("--rougeL", type=float, nargs=3,
                        help="Input ROUGE-L scores for Train, Validation, and Test sets.")

    args = parser.parse_args()

    # 如果提供了 BLEU 和 ROUGE 分数，则生成图表
    if args.bleu1 and args.bleu4 and args.rouge1 and args.rougeL:
        # 如果用户输入了 BLEU 和 ROUGE 分数，直接生成图表
        bleu1_scores = args.bleu1
        bleu4_scores = args.bleu4
        rouge1_scores = args.rouge1
        rougeL_scores = args.rougeL

        print("Generating charts with provided BLEU-1, BLEU-4 and ROUGE scores...")
        plot_scores(bleu1_scores, bleu4_scores, rouge1_scores, rougeL_scores)
        plot_score_table(bleu1_scores, bleu4_scores,
                         rouge1_scores, rougeL_scores)  # 生成表格图
    else:
        # 没有提供分数，则进行评估
        print("No scores provided, starting evaluation process...")

        # 评估数据集
        train_file_path = "dataset/flickr8k/train.txt"
        validation_file_path = "dataset/flickr8k/validation.txt"
        test_file_path = "dataset/flickr8k/test.txt"

        # 加载 ground truth
        train_ground_truths = load_ground_truths(train_file_path)
        validation_ground_truths = load_ground_truths(validation_file_path)
        test_ground_truths = load_ground_truths(test_file_path)

        # 提取所有图片ID
        train_image_ids = list(train_ground_truths.keys())
        validation_image_ids = list(validation_ground_truths.keys())
        test_image_ids = list(test_ground_truths.keys())

        # 分别评估训练集、验证集和测试集
        train_bleu1, train_bleu4, train_rouge = evaluate_dataset(
            train_image_ids, train_ground_truths, "Train")
        val_bleu1, val_bleu4, val_rouge = evaluate_dataset(
            validation_image_ids, validation_ground_truths, "Validation")
        test_bleu1, test_bleu4, test_rouge = evaluate_dataset(
            test_image_ids, test_ground_truths, "Test")

        # 生成并保存图表
        bleu1_scores = [train_bleu1, val_bleu1, test_bleu1]
        bleu4_scores = [train_bleu4, val_bleu4, test_bleu4]
        rouge1_scores = [train_rouge['rouge1'],
                         val_rouge['rouge1'], test_rouge['rouge1']]
        rougeL_scores = [train_rouge['rougeL'],
                         val_rouge['rougeL'], test_rouge['rougeL']]

        plot_scores(bleu1_scores, bleu4_scores, rouge1_scores, rougeL_scores)
        plot_score_table(bleu1_scores, bleu4_scores,
                         rouge1_scores, rougeL_scores)  # 生成表格图
