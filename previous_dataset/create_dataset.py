"""
既存研究「The Orienteering Problem with Stochastic Profits」のSet4をデータセットとする．
ノード数：25~150で25ずつ増やす
ノードの座標：ランダムな一様分布．x,y座標は0~100
ノードの報酬と分散は離散一様分布
報酬の平均の範囲：[10,20]
報酬の分散の範囲：[16,25], [225,400]
高分散と低分散の位置はランダム．
"""

import json
import math
import os
import random
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def clear_directory(directory: str) -> None:
    """
    指定されたディレクトリ内のjson, pngファイルを削除する関数。

    :param directory: 対象のディレクトリパス
    """
    path = Path(directory)
    if path.exists():
        # ディレクトリ内のjson, pngファイルを削除
        for file in path.glob("*.json"):
            file.unlink()  # ファイル削除
        for file in path.glob("*.png"):
            file.unlink()  # ファイル削除

def tsp_2opt(distances: List[List[float]]) -> Tuple[float, List[int]]:
    num_nodes = len(distances)

    # 初期経路を生成 (ノード0から始まり、終了ノードがノード最大番号)
    current_path = list(range(num_nodes))
    current_cost = calculate_path_cost(current_path, distances)

    improved = True
    while improved:
        improved = False
        for i in range(1, num_nodes - 2):  # スタートノード(0)と終了ノードは固定
            for j in range(i + 1, num_nodes - 1):
                # 新しい経路を生成 (部分逆転)
                new_path = current_path[:i] + current_path[i:j + 1][::-1] + current_path[j + 1:]
                new_cost = calculate_path_cost(new_path, distances)
                if new_cost < current_cost:
                    current_path = new_path
                    current_cost = new_cost
                    improved = True
                    break  # 改善があれば次のループに進む
            if improved:
                break

    return current_cost, current_path

def calculate_path_cost(path: List[int], distances: List[List[float]]) -> float:
    cost = 0
    for i in range(len(path) - 1):
        cost += distances[path[i]][path[i + 1]]
    return cost

def create_node_list(lower_node: int, upper_node: int, step: int) -> List[str]:
    if not all(isinstance(x, int) for x in [lower_node, upper_node, step]):
        raise ValueError("All inputs must be integers.")
    if step <= 0:
        raise ValueError("Step must be a positive integer.")
    if lower_node > upper_node:
        raise ValueError("lower_node must be less than or equal to upper_node.")

    number_list = np.arange(lower_node, upper_node + 1, step).tolist()
    node_list = [f"{num:03}" for num in number_list]
    return node_list

def create_dataset_list(dataset_count: int) -> List[str]:
    padded_list = [f"{i:03}" for i in range(1, dataset_count + 1)]
    return padded_list

def create_dataset_and_save() -> None:
    # ディレクトリ名
    directory = "previous_dataset"
    Path(directory).mkdir(exist_ok=True)

    clear_directory(directory)
    # 各ノード数のデータを作成
    for num_node_str in node_count_list:  # ノード数25~150
        for file_suffix in dataset_count_list:  # ファイル番号001~dataset_count
            # ファイル名を作成
            num_nodes = int(num_node_str)
            file_name_json = f"vertex_{num_node_str}_{file_suffix}.json"
            file_name_png = f"vertex_{num_node_str}_{file_suffix}.png"
            file_path_json = os.path.join(directory, file_name_json)
            file_path_png = os.path.join(directory, file_name_png)

            # ファイルが存在する場合は削除
            if os.path.exists(file_path_json):
                os.remove(file_path_json)
            if os.path.exists(file_path_png):
                os.remove(file_path_png)

            # ノード情報の生成
            vertices = []
            scores_ave = []
            scores_var = []

            # 最初のノードの座標
            x0 = random.uniform(0, 100)
            y0 = random.uniform(0, 100)
            vertices.append((x0, y0))

            # 最初のノードの報酬は0に設定
            scores_ave.append(0)
            scores_var.append(0)

            for _ in range(1, num_nodes - 1):
                # ノードの座標 (x, y)
                x = random.uniform(0, 100)
                y = random.uniform(0, 100)
                vertices.append((x, y))

                # 報酬の平均 (10～20)
                reward_mean = random.uniform(10, 20)
                scores_ave.append(reward_mean)

                # 報酬の分散 (高分散 or 低分散)
                if random.choice([True, False]):  # 高分散 or 低分散
                    reward_variance = random.uniform(225, 400)
                else:
                    reward_variance = random.uniform(16, 25)
                scores_var.append(reward_variance)

            # 最後のノードの座標を最初のノードと同じに設定
            vertices.append((x0, y0))

            # 最後のノードの報酬は0に設定
            scores_ave.append(0)
            scores_var.append(0)

            # ノード間の距離行列
            distances = []
            for i in range(num_nodes):
                distances_row = []
                for j in range(num_nodes):
                    dx = vertices[i][0] - vertices[j][0]
                    dy = vertices[i][1] - vertices[j][1]
                    distances_row.append(math.sqrt(dx**2 + dy**2))
                distances.append(distances_row)

            # 動的計画法で最小コストを計算
            min_cost, best_path = tsp_2opt(distances)

            # JSONデータの構築
            data = {
                "vertices": vertices,
                "distances": distances,
                "scores_ave": scores_ave,
                "scores_var": scores_var,
                "vertex_number": num_nodes,
                "minimum_distances": {
                    "cost": min_cost,
                    "path": best_path
                }
            }

            # JSONファイルとして保存
            with open(file_path_json, "w") as file:
                json.dump(data, file, indent=4)

            # グラフのプロット
            x_vals, y_vals = zip(*vertices)
            plt.figure()
            plt.scatter(x_vals, y_vals, c="blue", label="Nodes")
            plt.scatter(x0, y0, c="red", label="Start/End Node", s=100)  # 0番目と最終ノードを強調
            # 各ノードの座標に番号を追加
            for idx, (x, y) in enumerate(vertices):
                plt.text(x + 1, y + 1, str(idx), fontsize=8, color="black", ha='left', va='bottom')
            plt.title(f"Graph: {file_name_json}")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()
            plt.grid(True)
            plt.savefig(file_path_png)
            plt.close()

if __name__ == "__main__":
    dataset_count = 100  # ノードあたりのデータセット数
    lower_node = 15
    upper_node = 25
    step = 1
    node_count_list = create_node_list(lower_node, upper_node, step)
    dataset_count_list = create_dataset_list(dataset_count)
    create_dataset_and_save()