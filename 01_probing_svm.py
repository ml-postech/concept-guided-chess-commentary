import os
import sys
from os.path import *
import shutil
import importlib
import traceback

import yaml
import jsonlines
import pickle
import wandb
from datetime import datetime
import logging
import tqdm


from sklearn import svm
import numpy as np
import tensorflow as tf 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from stockfish import Stockfish
from lczeroTraining.tf.tfprocess import TFProcess
from lcztools import _leela_board as leelaBoard


def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")

def setup_logger(logger_name, root, level=logging.DEBUG, screen=False, tofile=False):
    """set up logger"""
    os.makedirs(root, exist_ok=True)
    lg = logging.getLogger(logger_name)
    lg.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    handlers = []
    if tofile:
        log_file = os.path.join(root, "_{}.log".format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        handlers.append(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        handlers.append(sh)
    logging.basicConfig(level=level, handlers=handlers)
    lg.info("Logger setup")
    return lg


tf_cfg_path = "./lczeroTraining/tf/configs/T78.yaml"
tf_ckp_path = "../lc0/T78-512x40-782344.pb.gz"
stockfish_8_path = "../stockfish-8-linux/Linux/stockfish_8_x64_modern"

mode = "train"          # ["train", "svm_train", "svm_eval"]
wandb_project = "lc0_concept"
name = "240824_02_negrand_large"
target_lrs = [1e-4, 1e-6]
# target_lrs = [1e-3, 1e-4, 1e-5, 1e-6]
num_epochs = 100

lichess_eval_path = "data/lichess_db_eval.jsonl"
sts_path = "./STS/STS1-STS15_LAN_v3.epd"
lichess_puzzle_path = './lichess_db_puzzle.csv'

data_size = 200000
data_ratio = 0.05
test_split = 0.1

concept_extraction_model = "linear_svm"     # ["linear_svm", "svm", "mlp"]
concept_extraction_version = "v4.6"     # 

cbm_data_size = 10000
batch_size = 4
# cbm_label = "stockfish"       # ["lc0", "stockfish"]
cbm_label = "lc0"       # ["lc0", "stockfish"]

target_layers = [39, 40]

stockfish_concepts = [
    "Material_t_mid", "Imbalance_t_mid", "Pawns_t_mid", 
    "Knights_w_mid", "Knights_b_mid",
    "Bishop_w_mid", "Bishop_b_mid",
    "Rooks_w_mid", "Rooks_b_mid",
    "Queens_w_mid", "Queens_b_mid",
    "Mobility_w_mid", "Mobility_b_mid",
    "Kingsafety_w_mid", "Kingsafety_b_mid",
    "Threats_w_mid", "Threats_b_mid",
    "Space_w_mid", "Space_b_mid",
    "Passedpawns_w_mid", "Passedpawns_b_mid",
]

sts_concepts = [
    "Undermining",
    "Open_Files_and_Diagonals",
    "Knight_Outposts",
    "Square_Vacancy",
    "Bishop_vs_Knight",
    "Re-Capturing",
    "Offer_of_Simplification",
    "Advancement_of_f/g/h_Pawns",
    "Advancement_of_a/b/c_Pawns",
    "Simplification",
    "Activity_of_the_King",
    "Center_Control",
    "Pawn_Play_in_the_Center",
    "Queens_and_Rooks_to_the_7th_rank",
    "Avoid_Pointless_Exchange",
]

puzzle_concepts = [
    "fork",
    "pin",
    "mate",
    # "defensiveMove",
    "hangingPiece",
    "sacrifice",
    "attraction",
    "deflection",
    "skewer",
    "discoveredAttack",
    "capturingDefender",
    "exposedKing",
    # "zugzwang",
]

target_concepts = stockfish_concepts
# target_concepts = stockfish_concepts + sts_concepts + puzzle_concepts
# target_concepts = [
#     "mate",
#     "sacrifice",
# ]


cfg = {
    "name": name,
    "target_concepts": target_concepts,
    "target_lrs": target_lrs,
    "target_layers": target_layers,
    "num_epochs": num_epochs,
    "concept_extraction_model": concept_extraction_model,
    "cbm_label": cbm_label,
}

result_dir = f"results/{name}"
os.makedirs(result_dir, exist_ok=True)
logger = setup_logger(logger_name=__name__, level=logging.INFO, root=result_dir, screen=True, tofile=True)
shutil.copy(__file__, f"{result_dir}/run_script.py") 

stockfish_engine = Stockfish(path=stockfish_8_path)

def load_tf_net(tf_cfg_path, tf_ckp_path):
    with open(tf_cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    tfproc = TFProcess(cfg)
    tfproc.init_net(return_activations=True)
    tfproc.replace_weights(tf_ckp_path)
    return tfproc

def fen_to_tf_input(fen):
    board = leelaBoard.LeelaBoard(fen=fen)
    input_np = np.reshape(board.lcz_features(), [1, 112, 8, 8])
    input_tf = tf.convert_to_tensor(input_np)
    return input_tf

def parse_eval(eval_str):
    result = {}
    body = eval_str.split("----------------+-------------+-------------+-------------")[1]
    lines = body.strip().split("\n")
    for line in lines:
        term = line.split("|")[0].strip()
        term = term.replace(" ", "")
        result[f"{term}_w_mid"] = line.split("|")[1].split()[0]
        result[f"{term}_w_end"] = line.split("|")[1].split()[1]
        result[f"{term}_b_mid"] = line.split("|")[2].split()[0]
        result[f"{term}_b_end"] = line.split("|")[2].split()[1]
        result[f"{term}_t_mid"] = line.split("|")[3].split()[0]
        result[f"{term}_t_end"] = line.split("|")[3].split()[1]
    return result

def get_concept_from_fen(fen):
    stockfish_engine.set_fen_position(fen)
    eval_str = stockfish_engine.get_static_eval_8()
    info = parse_eval(eval_str)
    return info


def get_fen_data(data_size=data_size):
    probing_fen_data = []
    
    with jsonlines.open(lichess_eval_path) as f:
        count = 0
        for line in f.iter():
            if count > data_size:
                break
            probing_fen_data.append(line['fen'] + " 0 0")
            count += 1
    
    return probing_fen_data


def get_data_for_stockfish_concept(fen_data, target_concept):

    scores = []
    for i, fen in enumerate(fen_data):
        eval_info = get_concept_from_fen(fen)
        score = eval_info[target_concept]
        scores.append((score, i))
    
    ss = sorted(scores)
    probing_data_size = int(data_size * data_ratio)
    train_data_size = int(probing_data_size * (1 - test_split))
    test_data_size = probing_data_size - train_data_size
    pos_data_t = ss[:probing_data_size]
    pos_data = list(map(lambda x: fen_data[x[1]], pos_data_t))
    neg_data_t = ss[-probing_data_size:]
    neg_data = list(map(lambda x: fen_data[x[1]], neg_data_t))

    train_data = pos_data[:train_data_size] + neg_data[:train_data_size]
    train_label = [1] * train_data_size + [0] * train_data_size
    test_data = pos_data[train_data_size:] + neg_data[train_data_size:]
    test_label = [1] * test_data_size + [0] * test_data_size

    return train_data, train_label, test_data, test_label

def get_data_for_sts_concept(target_concept):
    pos_data = []
    neg_data = []
    sts_idx = sts_concepts.index(target_concept)
    with open(sts_path, "r") as f:
        lines = f.readlines()
    sts_data = lines[sts_idx * 100:(sts_idx + 1) * 100]
    for line in sts_data:
        fen = line.split("bm")[0]
        candidate_moves_uci = line.split('"')[9].split(" ")
        best_move = candidate_moves_uci[0]

        stockfish_engine.set_fen_position(fen)
        top_moves = stockfish_engine.get_top_moves(5)
        bad_move_list = [m['Move'] for m in top_moves if m['Move'] not in candidate_moves_uci]
        bad_move = bad_move_list[0] if len(bad_move_list) > 0 else top_moves[-1]['Move']
        # bad_moves = bad_move_list[:2] if len(bad_move_list) > 1 else \
        #     [top_moves[-2]['Move'], top_moves[-1]['Move']] if len(top_moves) >= 2 else \
        #     [top_moves[-1]['Move'], top_moves[-1]['Move']]

        stockfish_engine.make_moves_from_current_position([best_move])
        best_reply = stockfish_engine.get_best_move()
        stockfish_engine.make_moves_from_current_position([best_reply])
        pos_fen = stockfish_engine.get_fen_position()

        stockfish_engine.set_fen_position(fen)
        stockfish_engine.make_moves_from_current_position([bad_move])
        best_reply = stockfish_engine.get_best_move()
        stockfish_engine.make_moves_from_current_position([best_reply])
        neg_fen = stockfish_engine.get_fen_position()


        pos_data.append(pos_fen)
        neg_data.append(neg_fen)
    
    train_data_size = int(len(pos_data) * (1 - test_split))
    test_data_size = len(pos_data) - train_data_size
    
    train_data = pos_data[:train_data_size] + neg_data[:train_data_size]
    train_label = [1] * train_data_size + [0] * train_data_size
    test_data = pos_data[train_data_size:] + neg_data[train_data_size:]
    test_label = [1] * test_data_size + [0] * test_data_size
    return train_data, train_label, test_data, test_label

def get_data_for_puzzle_concept(target_concept, max_num):
    pos_data = []
    neg_data = []
    puzzle_data = []
    puzzle_neg_data = []
    with open(lichess_puzzle_path, "r") as f:
        f.readline()
        for line in f:
            if "mateIn1" in line:
                continue
            if target_concept in line:
                puzzle_data.append(line)
            else:
                puzzle_neg_data.append(line)
            if len(puzzle_data) >= max_num and len(puzzle_neg_data) >= max_num:
                break
    puzzle_data = puzzle_data[:max_num]
    puzzle_neg_data = puzzle_neg_data[:max_num]
    for line, line_neg in tqdm.tqdm(zip(puzzle_data, puzzle_neg_data)):
        _, base_fen, moves, rating, _, _, _, tags, *etc = line.split(",")
        stockfish_engine.set_fen_position(base_fen)
        stockfish_engine.make_moves_from_current_position([moves.split()[0]])
        fen = stockfish_engine.get_fen_position()

        _, neg_fen, _, _, _, _, _, tags, *etc = line_neg.split(",")

        pos_data.append(fen)
        neg_data.append(neg_fen)


    
    train_data_size = int(len(pos_data) * (1 - test_split))
    test_data_size = len(pos_data) - train_data_size
    
    train_data = pos_data[:train_data_size] + neg_data[:train_data_size]
    train_label = [1] * train_data_size + [0] * train_data_size
    test_data = pos_data[train_data_size:] + neg_data[train_data_size:]
    test_label = [1] * test_data_size + [0] * test_data_size
    return train_data, train_label, test_data, test_label


def get_data_for_concept(fen_data, target_concept):
    if target_concept in sts_concepts:
        return get_data_for_sts_concept(target_concept)
    elif target_concept in puzzle_concepts:
        return get_data_for_puzzle_concept(target_concept, max_num=int(data_size * data_ratio))
    else:
        return get_data_for_stockfish_concept(fen_data, target_concept)



def concept_linear_probing(tfproc, target_layer, train_set_fen, train_set_label, test_set_fen, test_set_label):
    xs = []
    train_zs = []
    for fen in tqdm.tqdm(train_set_fen):
        #TODO: batch
        x = fen_to_tf_input(fen)
        xs.append(x)
        z = tfproc.activation_extractor.predict(x)
        train_zs.append(z[target_layer].reshape(-1))
    train_zs = np.stack(train_zs)

    labels = train_set_label

    if concept_extraction_model == "linear_svm":
        clf = svm.LinearSVC()
    else:
        clf = svm.SVC()
    clf.fit(train_zs + (np.random.randn(*train_zs.shape) - 0.5) * 0.1, labels)

    xs = []
    test_zs = []
    for fen in tqdm.tqdm(test_set_fen):
        x = fen_to_tf_input(fen)
        xs.append(x)
        z = tfproc.activation_extractor.predict(x)
        test_zs.append(z[target_layer].reshape(-1))
    test_zs = np.stack(test_zs)

    test_labels = test_set_label

    train_pred = clf.predict(train_zs)
    test_pred = clf.predict(test_zs)
    train_acc = 1 - np.average(np.abs(train_pred - np.array(labels)))
    train_precision = np.sum(np.logical_or(train_pred, np.array(labels)) == 0) / np.sum(train_pred == 0)
    train_recall = np.sum(np.logical_or(train_pred, np.array(labels)) == 0) / np.sum(np.array(labels) == 0)
    test_acc = 1 - np.average(np.abs(test_pred - np.array(test_labels)))
    test_precision = np.sum(np.logical_or(test_pred, np.array(test_labels)) == 0) / np.sum(test_pred == 0)
    test_recall = np.sum(np.logical_or(test_pred, np.array(test_labels)) == 0) / np.sum(np.array(test_labels) == 0)
    
    return clf, train_acc, train_precision, train_recall, test_acc, test_precision, test_recall


def eval_linear_probing(clf, tfproc, target_layer, test_set_fen, test_set_label):

    xs = []
    test_zs = []
    for fen in tqdm.tqdm(test_set_fen):
        x = fen_to_tf_input(fen)
        xs.append(x)
        z = tfproc.activation_extractor.predict(x)
        test_zs.append(z[target_layer].reshape(-1))
    test_zs = np.stack(test_zs)

    test_labels = test_set_label

    test_pred = clf.predict(test_zs)
    test_acc = 1 - np.average(np.abs(test_pred - np.array(test_labels)))
    test_precision = np.sum(np.logical_or(test_pred, np.array(test_labels)) == 0) / np.sum(test_pred == 0)
    test_recall = np.sum(np.logical_or(test_pred, np.array(test_labels)) == 0) / np.sum(np.array(test_labels) == 0)
    
    return test_acc, test_precision, test_recall


def activation_extraction(tfproc, target_layer, fen_data):
    xs = []
    train_zs = []
    for fen in tqdm.tqdm(fen_data, desc="activation"):
        #TODO: batch
        x = fen_to_tf_input(fen)
        xs.append(x)
        z = tfproc.activation_extractor.predict(x)
        train_zs.append(z[target_layer].reshape(-1))

    return np.stack(train_zs)

def get_stockfish_label(fen_data):
    scores = []
    for fen in tqdm.tqdm(fen_data, desc="stockfish eval"):
        stockfish_engine.set_fen_position(fen)
        eval = stockfish_engine.get_static_eval()
        if eval != None:
            eval = eval / 100
        scores.append(eval)
    return scores


def get_lc0_label(fen_data, tfproc):
    scores = []
    bs = 10
    for i in tqdm.tqdm(range(len(fen_data) // bs), desc="lc0 eval"):
        inputs = []
        for fen in fen_data[bs * i: bs * (i + 1)]:
            input_tf = fen_to_tf_input(fen)
            inputs.append(input_tf)
        input_tf = np.concatenate(inputs, axis=0)

        policy, wdl, move_left = tfproc.model.predict(input_tf)

        for w in wdl:
            scores.append(w)
    return scores


class Regressor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__() 
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True) 
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=True) 

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def prepare_dataset(fen_dataset, tfproc, svms, target_layer):
    train_activations = activation_extraction(tfproc, target_layer, fen_dataset) # TODO
    train_concepts = []
    for key, clf in tqdm.tqdm(svms.items(), desc="concepts"):
        train_concept = clf.decision_function(train_activations)
        train_concepts.append(train_concept)
    train_concepts_np = np.stack(train_concepts).T
    train_concepts_np = np.clip(train_concepts_np, -2, 2)
    if cbm_label == "lc0":
        train_labels = get_lc0_label(fen_dataset, tfproc)
        train_labels = np.stack(train_labels)
    elif cbm_label == "stockfish":
        train_labels = get_stockfish_label(fen_dataset)
        train_labels = np.array(train_labels)
        removes = np.where(train_labels == None)[0]
        train_concepts_np = np.delete(train_concepts_np, removes, axis=0)
        train_labels = np.delete(train_labels, removes, axis=0)
    else:
        raise Exception("Unimplemented CBM label")

    dataset = TensorDataset(torch.Tensor(train_concepts_np), torch.Tensor(train_labels))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataset, dataloader

def main():
    tfproc = load_tf_net(tf_cfg_path, tf_ckp_path)

    loss_func = nn.MSELoss()
    wandb.init(project=wandb_project, config=cfg, name=name)

    summary = {}
    clfs = {target_layer: {} for target_layer in target_layers}

    fen_data = get_fen_data(data_size=data_size)
    for target_layer in target_layers:
        for target_concept in target_concepts:
            key = f"size_{data_size}_{data_ratio}_concept_{target_concept}_layer_{target_layer}".replace("/", "_")
            cache_path = f"cache/{concept_extraction_model}_{concept_extraction_version}_{key}.pkl"
            if os.path.exists(cache_path) and mode != "svm_train":
                with open(cache_path, "rb") as f:
                    clf = pickle.load(f)
                logger.info(f"loaded from {cache_path}")
                if mode == "svm_eval":
                    train_set_fen, train_set_tag, test_set_fen, test_set_tag = get_data_for_concept(fen_data, target_concept)    
                    test_acc, test_pr, test_recall, = eval_linear_probing(clf, tfproc, target_layer, test_set_fen, test_set_tag)
                    summary[key] = (test_acc, test_pr, test_recall)
            else:
                train_set_fen, train_set_tag, test_set_fen, test_set_tag = get_data_for_concept(fen_data, target_concept)
                clf, train_acc, train_pr, train_recall, test_acc, test_pr, test_recall, = concept_linear_probing(tfproc, target_layer, train_set_fen, train_set_tag, test_set_fen, test_set_tag)
                summary[key] = (train_acc, train_pr, train_recall, test_acc, test_pr, test_recall)
                logger.info(f"{key} : {(train_acc, train_pr, train_recall, test_acc, test_pr, test_recall)}")
                with open(cache_path, "wb") as f:
                    pickle.dump(clf, f)
            clfs[target_layer][key] = clf

    logger.info(summary)

    if mode in ["train"]:
        cbm_train_data = fen_data[:cbm_data_size]
        cbm_test_data = fen_data[cbm_data_size:int(cbm_data_size * 1.1)]
        num_concepts = len(clfs[target_layers[0]].keys())

        for target_layer in target_layers:
            train_dataset, train_dataloader = prepare_dataset(cbm_train_data, tfproc, clfs[target_layer], target_layer)
            test_dataset, test_dataloader = prepare_dataset(cbm_test_data, tfproc, clfs[target_layer], target_layer)

            logger.info("dataset ready")
            for lr in target_lrs:

                out_dim = 3 if cbm_label == "lc0" else 1
                predictor_model = Regressor(num_concepts, out_dim).cuda()
                optim = torch.optim.Adam(predictor_model.parameters(), lr=lr) # TODO
                loss_func = nn.MSELoss()

                step = 0
                for epoch in range(num_epochs):
                    losses = []
                    for z, y in train_dataloader:
                        z = z.cuda()
                        y = y.cuda()
                        loss = loss_func(predictor_model(z), y)

                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        losses.append(loss.item())
                        step += 1
                    avg_loss = sum(losses) / len(losses)
                    wandb.log({
                        "epoch": epoch,
                        "step": step,
                        "lr": lr,
                        "train_loss": avg_loss,
                        f"train_loss_{lr}_l{target_layer}": avg_loss,
                    })
                    logger.info(f"epoch: {epoch}, loss: {avg_loss}")

                    
                    losses = []
                    with torch.no_grad():
                        for z, y in test_dataloader:
                            z = z.cuda()
                            y = y.cuda()
                            loss = loss_func(predictor_model(z), y)

                            losses.append(loss.item())
                    avg_loss = sum(losses) / len(losses)
                    wandb.log({
                        "epoch": epoch,
                        "lr": lr,
                        "valid_loss": avg_loss,
                        f"valid_loss_{lr}_l{target_layer}": avg_loss,
                    })
                    logger.info(f"epoch: {epoch}, test_loss: {avg_loss}")

                file_name = f"{result_dir}/cbm_{lr}_l{target_layer}.pkl"
                torch.save(predictor_model.state_dict(), file_name)

        logger.info("end")


if __name__ == "__main__":
    main()