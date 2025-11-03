# LLM-jp NeMo-RL

```
llm_jp/
├── configs/          # 設定ファイル等
├── scripts/          # 各種スクリプト (ジョブスクリプト等)
├── environment.sh    # ABCI 3.0 用の環境設定スクリプト
└── README.md
```

## 開発方針

- 基本的に変更は `llm_jp/` 以下に集約し、元の NeMo-RL リポジトリからの変更点を最小限に抑える。
- 元の NeMo-RL リポジトリの更新を容易に取り込めるようにする。
  - `llm_jp/` 以外のファイルへの変更を行う場合は、pull requestを作成して、変更内容を明確に記載する。
- LLM-jp 特有の変更点はこのドキュメントに記録する。

## 環境構築

```bash
git clone https://github.com/llm-jp/nemo-rl.git
cd nemo-rl
git submodule update --init --recursive

source llm_jp/environment.sh
uv venv --python=3.12
uv sync
```
