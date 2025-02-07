# ruff: noqa: F401
import json
import os
import re
import statistics
import sys
import time
from typing import List, Tuple

import japanize_matplotlib  # 日本語化ライブラリ
import matplotlib.pyplot as plt
from cef_alpha_step import ConvexEllipse as ConvexEllipseAlpha
from exact_cef import CEF_DP


class GetOutPut:
    def __init__(self, member, t_max_ratio, probability):
        self.member = member
        self.t_max_ratio = t_max_ratio
        self.probability = probability
        self.file_path = "graph_plot/compare/"

    def main_compare_cef(self):
        vertex_num_list, dataset_list = self.get_numbers_from_previous_directory("previous_dataset")
        for vertex_num in vertex_num_list:
            for dataset in dataset_list:
                print(f"{vertex_num=}, {dataset=}")
                self.proposed_cef_ellipse_method_alpha001(vertex_num, dataset, 20, 5)
                self.proposed_cef_ellipse_method_alpha01(vertex_num, dataset, 20, 5)
                self.cef_by_dp(vertex_num, dataset)
                
        
        print("\n\n\n\n~~~~~~~~~finish~~~~~~~~")

    def proposed_cef_ellipse_method_alpha001(self, vertex_num: str, file_id: str, outer_loop: int, inner_loop: int) -> Tuple[List[float], List[float]]:
        """提案手法の楕円法でCEFを構成する方法を用いてチームスコアを算出する．alpha=0.01刻みで実行する

        Args:
            vertex_num (str): ノード数
            file_id (str): ファイルID
            outer_loop (int): 外側のループ回数
            inner_loop (int): 内側のループ回数
        Returns:
            Tuple[List[float], List[float]]: 同一ノード数におけるチームスコアのリスト，同一ノード数における実行時間のリスト
        """
        start_time = time.time()
        print("提案手法の楕円法でCEFを構成する方法（alpha=0.01）")
        proposed_ellipse_method = ConvexEllipseAlpha(vertex_num, file_id, self.member, self.t_max_ratio, self.probability, 100, outer_loop, inner_loop)
        team_score, cef = proposed_ellipse_method.main_exact_algorithm()
        end_time = time.time()
        exe_time = end_time - start_time
        self.update_experiment_results("Proposed Ellipse CEF alpha hundred", team_score, vertex_num, file_id, exe_time, cef, "compare_cef_method.json")
        self.plot_cef(vertex_num, file_id, cef, "Proposed CEF Step 001")
    
    def proposed_cef_ellipse_method_alpha01(self, vertex_num: str, file_id: str, outer_loop: int, inner_loop: int) -> Tuple[List[float], List[float]]:
        """提案手法の楕円法でCEFを構成する方法を用いてチームスコアを算出する．alpha=0.01刻みで実行する

        Args:
            vertex_num (str): ノード数
            file_id (str): ファイルID
            team_score_list (List[float]): 同一ノード数におけるチームスコアのリスト
            exe_time_list (List[float]): 同一ノード数における実行時間のリスト

        Returns:
            Tuple[List[float], List[float]]: 同一ノード数におけるチームスコアのリスト，同一ノード数における実行時間のリスト
        """
        start_time = time.time()
        print("提案手法の楕円法でCEFを構成する方法（alpha=0.1）")
        proposed_ellipse_method = ConvexEllipseAlpha(vertex_num, file_id, self.member, self.t_max_ratio, self.probability, 10, outer_loop, inner_loop)
        team_score, cef = proposed_ellipse_method.main_exact_algorithm()
        end_time = time.time()
        exe_time = end_time - start_time
        self.update_experiment_results("Proposed Ellipse CEF alpha one", team_score, vertex_num, file_id, exe_time, cef, "compare_cef_method.json")
        self.plot_cef(vertex_num, file_id, cef, "Proposed CEF Step 01")
    
    def cef_by_dp(self, vertex_num: str, file_id: str) -> Tuple[List[float], List[float]]:
        """DPを用いてCEFを構成する方法を用いる．

        Args:
            vertex_num (str): _description_
            file_id (str): _description_
            team_score_list (List[float]): _description_
            exe_time_list (List[float]): _description_

        Returns:
            Tuple[List[float], List[float]: _description_
        """
        start_time = time.time()
        print("提案手法の楕円法でCEFを構成する方法（alpha=0.1）")
        proposed_ellipse_method = CEF_DP(file_id, vertex_num, self.t_max_ratio, self.member, self.probability)
        team_score, cef = proposed_ellipse_method.main_exact_algorithm()
        end_time = time.time()
        exe_time = end_time - start_time
        self.update_experiment_results("CEF BY DP", team_score, vertex_num, file_id, exe_time, cef, "compare_cef_method.json")
        self.plot_cef(vertex_num, file_id, cef, "cef_by_dp")
    
    def update_experiment_results(self, name: str, team_score: float, vertex_num: int, file_id: str, exe_time: float, cef: List[List[float]], file_name: str) -> None:
        """jsonファイルに実験結果を追加または更新する関数。

        Args:
            name (str): _description_
            team_score (float): _description_
            vertex_num (int): _description_
            file_id (str): _description_
            exe_time (float): _description_
            cef (List[List[float]]): _description_
            file_path (str): _description_
        """
        # Path to the JSON file
        file_path = self.file_path + file_name
        # Load existing data from JSON file if it exists
        if os.path.exists(file_path):
            if os.path.getsize(file_path) > 0:  # Check if file is not empty
                with open(file_path, "r") as f:
                    data = json.load(f)
            else:
                data = []  # Empty file, initialize with empty list
        else:
            data = []


        # Create the new record
        new_record = {
            "name": name,
            "team_score": team_score,
            "exe_time": exe_time,
            "cef": cef,
            "member": self.member,
            "t_max_ratio": self.t_max_ratio,
            "probability": self.probability,
            "vertex_num": vertex_num,
            "file_id": file_id,
        }

        # Check for an existing record with the same vertex_num and file_id
        existing_record = next(
            (record for record in data if record.get("name")==name and record.get("vertex_num") == vertex_num and record.get("file_id") == file_id and record.get("member") == self.member and record.get("t_max_ratio") == self.t_max_ratio and record.get("probability") == self.probability), None
        )

        if existing_record:
            # Update the existing record
            existing_record.update(new_record)
        else:
            # Add the new record to the data list
            data.append(new_record)

        # Save the updated data back to the JSON file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    def plot_cef(self, vertex_num: str, file_id: str, cef: List[List[float]], name: str) -> None:
        """
        プロットして画像として保存する関数。

        Args:
            vertex_num (str): ノード数
            file_id (str): データセットのID
            cef (List[List[int]]): 実行可能解の集合の疑似端点
        """
        # x軸 (m) と y軸 (v) の値を抽出
        m_values = [point[0] for point in cef]
        v_values = [point[1] for point in cef]

        # m_values を基準にソートし、それに対応する v_values も並び替え
        sorted_points = sorted(zip(m_values, v_values), key=lambda x: x[0])
        sorted_m_values, sorted_v_values = zip(*sorted_points)

        # 保存パスの設定
        file_name = f"{name}_vertex_{vertex_num}_{file_id}_{self.member}_.png"
        save_path = os.path.join("graph_plot", "cef", file_name)

        # グラフをプロット
        plt.figure(figsize=(8, 6))
        plt.scatter(sorted_m_values, sorted_v_values, color="blue")
        plt.plot(sorted_m_values, sorted_v_values, color="dodgerblue", linewidth=0.6, linestyle="-")

        plt.xlabel("通過したノードの報酬の平均の総和$m_y$")
        plt.ylabel("通過したノードの報酬の分散の総和$v_y$")
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
        plt.legend()
        plt.grid()

        # グラフをPNGファイルに保存
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.show()
        
    
    def draw_graph(self, title: str, vertex_num_list: List[int], previous_ellipse: List[float], proposed_ellipse: List[float], proposed_ellipse_cef: List[float]) -> None:
        """
        チームスコアをグラフに描画する．手法全てを比較できるようにするために，バーには手法名を記載する．

        Args:
            title str: 何についてのグラフか
            vertex_num_list List[int]: 頂点数のリスト
            previous_ellipse List[float]: 既存研究の楕円法のリスト
            proposed_ellipse List[float]: 提案手法の楕円法のリスト
            proposed_ellipse_cef List[float]: 提案手法の楕円法でCEFを構成する方法のリスト
        """
        # グラフ保存先のディレクトリを定義
        save_dir = "graph_plot/compare"
        os.makedirs(save_dir, exist_ok=True)
        if title == "チームスコア":
            file = "team_score"
        elif title == "実行時間":
            file = "time"

        # ファイル名を設定
        file_name = f"compare_{file}_{self.member}.png"
        file_path = os.path.join(save_dir, file_name)

        # グラフを作成
        plt.figure(figsize=(10, 6))
        plt.plot(vertex_num_list, previous_ellipse, label="previous_ellipse", marker="o")
        plt.plot(vertex_num_list, proposed_ellipse, label="Proposed Ellipse", marker="s", linestyle="--")
        plt.plot(vertex_num_list, proposed_ellipse_cef, label="Proposed Ellipse CEF", marker="x", linestyle="-.")

        # グラフのタイトルとラベルを設定
        plt.title(f"種法における{title}の比較", fontsize=14)
        plt.xlabel("ノード数", fontsize=12)
        plt.ylabel(title, fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)

        # グラフを保存
        plt.savefig(file_path)
        plt.close()

        print(f"Graph saved to: {file_path}")
    
    def draw_graph_proposed_ellipse_cef(self, title: str, vertex_num_list: List[int], proposed_cef_ellipse: List[float]) -> None:
        """
        チームスコアをグラフに描画する．手法全てを比較できるようにするために，バーには手法名を記載する．

        Args:
            title str: 何についてのグラフか
            vertex_num_list List[int]: 頂点数のリスト
            proposed_cef_ellipse List[float]: 提案手法の楕円法によるチームスコア
        """
        # グラフ保存先のディレクトリを定義
        save_dir = "graph_plot/cef"
        os.makedirs(save_dir, exist_ok=True)
        if title == "チームスコア":
            file = "team_score"
        elif title == "実行時間":
            file = "time"

        # ファイル名を設定
        file_name = f"compare_{file}_{self.member}.png"
        file_path = os.path.join(save_dir, file_name)

        # グラフを作成
        plt.figure(figsize=(10, 6))
        plt.plot(vertex_num_list, proposed_cef_ellipse, label="Proposed Ellipse CEF", marker="o", linestyle="--")

        # グラフのタイトルとラベルを設定
        plt.title(f"種法における{title}の比較", fontsize=14)
        plt.xlabel("ノード数", fontsize=12)
        plt.ylabel(title, fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)

        # グラフを保存
        plt.savefig(file_path)
        plt.close()

        print(f"Graph saved to: {file_path}")
        
    def get_numbers_from_previous_directory(self, directory_path: str)    -> None:
        """
        指定されたディレクトリ内のファイル名から最初の3桁と最後の3桁の数字を抽出して別々のリストに格納
        また、リスト内の要素は一意で昇順にソート
        Args:
            directory_path (str): データセットのパス

        Returns:
            tuple: 2つのリスト（ノード数，データセット番号）
        """
        node_list = []
        dataset_list = []
        # 命名規則を正規表現で定義
        pattern = re.compile(r"vertex_(\d{3})\d*_(\d{3})\.json")

        # ディレクトリ内のファイルを走査
        for file_name in os.listdir(directory_path):
            match = pattern.match(file_name)
            if match:
                try:
                    # 正規表現から最初の3桁と最後の3桁を取得
                    node_number = match.group(1)
                    dataset_number = match.group(2)
                    node_list.append(node_number)
                    dataset_list.append(dataset_number)
                except ValueError as e:
                    print(f"Error processing {file_name}: {e}")

        # 要素を一意にして昇順にソート
        node_list = sorted(set(node_list))
        dataset_list = sorted(set(dataset_list))

        return node_list, dataset_list
        
    


if __name__ == "__main__":
    t_max_ratio = 0.3
    member_list = list(range(2, 5))
    member_list = [2]
    probability = 0.8
    for member in member_list:
        out_put = GetOutPut(member, t_max_ratio, probability)
        # team_paths = out_put.proposed_greedy_search_100("030", "001", [], [], 20, 5)
        out_put.main_compare_cef()
        # out_put.main_proposed_convex_ellipse()
