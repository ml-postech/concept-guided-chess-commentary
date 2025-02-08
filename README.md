# Concept-guided Chess Commentary Generation

This is a official repository for "Bridging the Gap between Expert and Language Models: Concept-guided Chess Commentary Generation and Evaluation" ([https://arxiv.org/abs/2410.20811](https://arxiv.org/abs/2410.20811)).

The following settings are tested on Ubuntu 20.04.

## 1. Environment setting

```
conda install python=3.7
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
<!-- conda install torch=1.8.2 -->
pip install tensorflow-gpu==2.5
pip install cairosvg python-chess==0.25
pip install PyQt5 PyQt5-tools PyQtWebEngine lxml nltk
pip install wandb scikit-learn
pip install jupyter notebook pyyaml jsonlines tqdm

pip install -e ./stockfish-py
```

```
export PYTHONPATH="$(readlink -f ./lczeroTraining/tf):$(readlink -f ./stockfish-py/):$(readlink -f ./lcztools/):$PYTHONPATH"
```

## 2. Data and model preparation

* ChessCommentary (Gameknot) dataset
  * Read and follow `gameknot_crawler/README.md`
  * If you follow the instructions, you will have `gameknot_crawler/saved_files/train.che-eng.single.che` and `gameknot_crawler/saved_files/train.che-eng.single.en`.
  * In `gameknow_crawler`, run `python data_converter.py`
  * It will create `gameknot_crawler/saved_files/train_single.pkl`

* Lichess Evaluations dataset 

  * [https://database.lichess.org/#evals]
  * Place `lichess_db_eval.jsonl` under `./data/`

* Stockfish 8
  * [https://drive.google.com/drive/folders/1nzrHOyZMFm4LATjF5ToRttCU0rHXGkXI]

* LeelaChessZero T78
  * [http://training.lczero.org/get_network?sha=d0ed346c32fbcc9eb2f0bc7e957d188c8ae428ee3ef7291fd5aa045fc6ef4ded]

## 3. Concept vector extraction

* Update paths and setting
  * stockfish_8_path
  * tf_ckp_path : lc0 T78 path
  * sts_path, lichess_puzzle_path : for other concepts

```
python 01_probing_svm.py
```
This will create `cache/*.pkl`

## 4. Concept-guided chess commentary generation

Update `02_ccc_generation.ipynb`
* Add OPENAI_API_KEY in the notebook

## 5. Automated chess commentary evaluation

Update `03_gcc_eval.ipynb`
* Add OPENAI_API_KEY in the notebook
* Read saved comments from log file and evaluate using gcc

## Note
The following codes are modified from the original repositories below.
* Gameknot crawler: [https://github.com/harsh19/ChessCommentaryGeneration].
* stockfish-py: [https://github.com/py-stockfish/stockfish].
* lczero-training: [https://github.com/LeelaChessZero/lczero-training].
* lcztools: [https://github.com/so-much-meta/lczero_tools]