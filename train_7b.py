"""
7B 模型训练脚本 - 用于远程服务器
"""
import torch
import gc
import json
import os
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.trainer import LMHeadOnlyTrainer, CrossBatchTrainer, SQuADDataset
from src.cross_batch_attention import CrossBatchAttention

# 配置
MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
DEVICE = 'cuda'
MAX_SAMPLES = 2000
NUM_EPOCHS = 3
BATCH_SIZE = 2  # 7B 模型用小 batch

def main():
    gc.collect()
    torch.cuda.empty_cache()

    print('=' * 60)
    print(f'训练配置: {MODEL_NAME}')
    print(f'样本数: {MAX_SAMPLES}, Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}')
    print('=' * 60)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据集
    train_dataset = SQuADDataset(tokenizer=tokenizer, split='train', max_samples=MAX_SAMPLES)
    print(f'训练数据集大小: {len(train_dataset)}')

    # 1. 训练 Baseline
    print('\n[1/2] 训练 Baseline (只训练 lm_head)')
    print('-' * 40)
    model_baseline = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    trainer_baseline = LMHeadOnlyTrainer(
        model=model_baseline,
        tokenizer=tokenizer,
        device=DEVICE,
        learning_rate=1e-4,
    )
    history_baseline = trainer_baseline.train(
        train_dataset=train_dataset,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        save_dir='checkpoints_baseline_7b',
    )
    print(f'Baseline 最终 Loss: {history_baseline["train_loss"][-1]:.4f}')

    del model_baseline, trainer_baseline
    gc.collect()
    torch.cuda.empty_cache()

    # 2. 训练 Cross-Batch
    print('\n[2/2] 训练 Cross-Batch (lm_head + cross-batch)')
    print('-' * 40)
    model_crossbatch = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    cross_batch_module = CrossBatchAttention(hidden_size=model_crossbatch.config.hidden_size)
    trainer_crossbatch = CrossBatchTrainer(
        model=model_crossbatch,
        tokenizer=tokenizer,
        cross_batch_module=cross_batch_module,
        device=DEVICE,
        learning_rate=1e-4,
        train_lm_head=True,
    )
    history_crossbatch = trainer_crossbatch.train(
        train_dataset=train_dataset,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        save_dir='checkpoints_crossbatch_7b',
    )
    print(f'Cross-Batch 最终 Loss: {history_crossbatch["train_loss"][-1]:.4f}')

    # 保存训练历史
    summary = {
        'config': {
            'model': MODEL_NAME,
            'train_samples': MAX_SAMPLES,
            'epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
        },
        'baseline_loss': history_baseline['train_loss'],
        'crossbatch_loss': history_crossbatch['train_loss'],
    }

    os.makedirs('outputs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'outputs/training_7b_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n' + '=' * 60)
    print('训练完成!')
    print(f'Baseline Loss: {history_baseline["train_loss"][-1]:.4f}')
    print(f'Cross-Batch Loss: {history_crossbatch["train_loss"][-1]:.4f}')
    print('=' * 60)

if __name__ == '__main__':
    main()
