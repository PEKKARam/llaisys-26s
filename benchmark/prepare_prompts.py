import argparse
import json
import shutil
from pathlib import Path

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_PATH = REPO_ROOT / "cache" / "prompt_sets" / "wikitext2_smoke.jsonl"

# 本地数据快照：第一次联网下载后保存在这里，之后直接从这里读取。
SOURCE_DIR = REPO_ROOT / "cache" / "datasets" / "wikitext2_raw_v1_test"
# 元信息文件最后才写入，它存在就说明快照是完整的。
META_PATH = REPO_ROOT / "cache" / "datasets" / "wikitext2_raw_v1_test.meta.json"

DATASET_REPO = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
PROMPT_LENGTHS = (16, 64, 128)
NUM_SOURCE_ROWS = 3


def load_source(refresh: bool):
    """优先读取本地快照；只有快照不存在（或指定 --refresh）时才联网。"""
    if not refresh and SOURCE_DIR.exists() and META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        dataset = load_from_disk(str(SOURCE_DIR))
        print("使用本地数据快照（未联网）")
        return dataset, meta["revision"]

    # 只有需要联网时才导入并使用 HfApi。
    from huggingface_hub import HfApi

    print("本地没有快照，正在联网下载……")
    revision = HfApi().dataset_info(DATASET_REPO).sha
    dataset = load_dataset(
        DATASET_REPO,
        DATASET_CONFIG,
        split="test",
        revision=revision,
    )

    # 清掉可能残留的不完整快照，再重新保存。
    if SOURCE_DIR.exists():
        shutil.rmtree(SOURCE_DIR)
    SOURCE_DIR.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(SOURCE_DIR))

    META_PATH.write_text(
        json.dumps(
            {
                "repo": DATASET_REPO,
                "config": DATASET_CONFIG,
                "split": "test",
                "revision": revision,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"快照已保存：{SOURCE_DIR}")
    return dataset, revision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="忽略本地快照，重新联网下载最新版本的数据集",
    )
    args = parser.parse_args()

    dataset, revision = load_source(args.refresh)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    print("数据集版本：", revision)
    print("记录数量：", len(dataset))

    samples = []
    selected_rows = 0
    minimum_tokens = max(PROMPT_LENGTHS)

    # enumerate 按原顺序遍历，row_id 是原数据集从 0 开始的行号。
    for row_id, row in enumerate(dataset):
        text = row["text"].strip()
        if not text or text.startswith("="):
            continue  # 跳过空行和 WikiText 的标题行。

        # 原文续写：不添加聊天模板，也不自动加入特殊 token。
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < minimum_tokens:
            continue  # 只有至少 128 个 token 的正文才能生成三个长度的样本。

        if any(token_id < 0 or token_id >= len(tokenizer) for token_id in ids):
            raise ValueError(f"第 {row_id} 行包含超出 tokenizer 词表范围的 token ID")

        for length in PROMPT_LENGTHS:
            samples.append({
                "id": f"row_{row_id}_len_{length}",
                "dataset_revision": revision,
                "source_row": row_id,
                "prompt_tokens": length,
                "input_ids": ids[:length],
                "input_mode": "raw_completion",
            })

        selected_rows += 1
        print(
            f"选中第 {row_id} 行：原文 {len(ids)} 个 token，"
            f"生成长度为 {PROMPT_LENGTHS} 的 3 条样本"
        )
        if selected_rows == NUM_SOURCE_ROWS:
            break

    # 数据不足时先报错，避免写出一份不完整的测试集。
    if selected_rows != NUM_SOURCE_ROWS:
        raise RuntimeError(
            f"需要 {NUM_SOURCE_ROWS} 条合格正文，实际只找到 {selected_rows} 条"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # 用 w 覆盖旧文件，重复执行不会不断追加相同样本。
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"已保存 {len(samples)} 条样本：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()