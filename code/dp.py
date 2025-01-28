# CEFアルゴリズムをDPで実装する．
# このPYファイルは動的手法の部分を格納している．
import json
import os
import re
import time
from itertools import combinations
from typing import List, Tuple

import scipy.stats as stats


class TOP_DP:
    def __init__(self, file_id: str, vertex_number: int, t_max: int, member: int, probability: float, alpha: float) -> None:
        self.vertex_num = vertex_number
        self.file_id = file_id
        self.vertices, self.vertices_distances, self.scores_ave, self.scores_var = self.load_json_data()
        self.vertex_num = int(vertex_number)
        self.t_max = t_max 
        self.member = member
        self.probability = probability
        self.c_p = -1 * stats.norm.ppf(probability)  # TODO 累積分布の逆関数の符号反転
        self.all_path = []
        self.start = 0
        self.goal = self.vertex_num - 1
        self.alpha = alpha
        
    def dp_main(self):
        INF = float('inf')
        # dp[S][i] を辞書型で定義
        dp = {}  # { (S, i): 累積移動時間 }
        prev = {}  # { (S, i): 親ノード }

        # 初期状態
        dp[(1 << self.start, self.start)] = 0  # 初期地点での累積移動時間は0

        # DPの遷移
        for S in range(1 << self.vertex_num):   # 全ての頂点について訪問、未訪問を表すビットマスクの全組み合わせ
            for i in range(self.vertex_num):    # 状態Sで訪問済み頂点iを探す
                if not (S & (1 << i)):  # 地点iが未訪問の場合スキップ
                    continue
                if (S, i) not in dp:  # 状態がまだ作られていない場合スキップ
                    continue

                for j in range(self.vertex_num):  # 頂点iから未訪問頂点jに移動する
                    if S & (1 << j):  # 地点jが訪問済みの場合スキップ
                        continue

                    # 次の移動のコスト
                    move_cost = self.vertices_distances[i][j]
                    if move_cost <= self.t_max:  # 制限時間内の移動のみ許可
                        new_total_cost = dp[(S, i)] + move_cost
                        if new_total_cost <= self.t_max:  # 累積移動時間が制限内である場合
                            new_state = (S | (1 << j), j)
                            if new_state not in dp or dp[new_state] > new_total_cost:
                                dp[new_state] = new_total_cost
                                prev[new_state] = i

        # 全ての実行可能なパスを洗い出す
        feasible_paths = []
        for (S, i) in dp.keys():
            if i == self.goal:  # ゴール地点に到達できる場合
                path = []
                current = i
                state = S
                while current != -1:
                    path.append(current)
                    next_state = state & ~(1 << current)  # 現在地を未訪問に戻す
                    current = prev.get((state, current), -1)
                    state = next_state
                path.reverse()  # スタート地点からの順序に直す
                feasible_paths.append(path)

        # opt_value, opt_paths = self.select_opt_path(feasible_paths)
        mean, var, opt_paths = self.select_opt_path(feasible_paths)
        return mean, var, opt_paths
                
    def select_opt_path(self, feasible_paths: List[List[int]]) -> Tuple[float, List[List[int]]]:  # type: ignore
        opt_value = -float('inf')
        opt_paths = []
        mean_tmp = 0
        variance_tmp = 0
        mean = 0
        variance = 0

        if self.member == 1:  # self.member が 1 の場合、組み合わせは不要
            for path in feasible_paths:
                visited_vertices = set(path)
                obj_value = self.calculate_obj_function(visited_vertices)

                # より良い目的関数値の場合、更新
                if obj_value > opt_value:
                    opt_value = obj_value
                    opt_paths = [path]  # 最適なパスを 1 次元配列として保存
        else:
            # 通常の場合、パスの組合せを取得
            for comb in combinations(feasible_paths, self.member):
                visited_vertices = set()
                for path in comb:
                    visited_vertices.update(path)  # 訪問頂点が被っていても1回としてカウント
                obj_value, mean_tmp, variance_tmp = self.calculate_obj_function(visited_vertices)

                # 目的関数値が最大の場合更新
                if obj_value > opt_value:
                    opt_value = obj_value
                    mean = mean_tmp
                    variance = variance_tmp
                    opt_paths = comb  # 最適な組み合わせのパスを保存
                    
        # return opt_value, opt_paths
        print(f"{opt_paths=}")
        return mean, variance, opt_paths
        
    def calculate_obj_function(self, visited_vertices: List[int]) -> float:
        mean = sum(self.scores_ave[v] for v in visited_vertices)
        variance = sum(self.scores_var[v] for v in visited_vertices)
        # obj_value = mean + self.c_p * math.sqrt(variance)
        obj_value = self.alpha * mean -  (1-self.alpha) * variance
        return obj_value, mean, variance

    def load_json_data(self) -> Tuple[List[List[int]], List[List[int]], List[int], int]:
        dataset_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../previous_dataset')
        json_files = [f for f in os.listdir(dataset_directory) if f.endswith('.json')]
        pattern = fr'vertex_{self.vertex_num}_{self.file_id}\.json$'
        print(f"{pattern=}")
        for file_name in json_files:
            if re.search(pattern, file_name):
                json_path = os.path.join(dataset_directory, file_name)
                with open(json_path, 'r') as json_file:
                    data = json.load(json_file)
                    self.vertives = data.get('vertices', [])
                    self.vertices_distances = data.get('distances', [])
                    self.scores_ave = data.get('scores_ave', [])
                    self.scores_var = data.get('scores_var', [])
                return self.vertives, self.vertices_distances, self.scores_ave, self.scores_var
        print(f"{self.vertex_num}を含むJSONファイルは見つかりませんでした。")
        return None, None, None, None
    
def write_json(opt_value: float, paths: List[int], exe_time: float) -> None:
    # 出力ディレクトリの作成
    output_dir = os.path.join("proposed_method", "cef_dp_result")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # JSONファイルのパス
    output_file = os.path.join(output_dir, "compare.json")
    # 実行時間を「時間」「分」「秒」で表現
    hours = int(exe_time // 3600)
    minutes = int((exe_time % 3600) // 60)
    seconds = exe_time % 60
    if hours > 0:
        exe_time_jpn = f"{hours}時間{minutes}分{seconds:.2f}秒"
    elif minutes > 0:
        exe_time_jpn = f"{minutes}分{seconds:.2f}秒"
    else:
        exe_time_jpn = f"{seconds:.2f}秒"
        
    # 新しいデータ
    new_entry = {
        "algorithm_name": "Dinamic Problem",
        "vertex_number": vertex_number,
        "file_id": file_id,
        "t_max": t_max,
        "member": member,
        "probability": probability,
        "opt_value": opt_value,
        "paths": paths,
        "exe_time": exe_time,
        "exe_time_jpn": exe_time_jpn
    }
    
    # ファイルが存在する場合は読み込み、存在しない場合は空のリストを初期化
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    
    # 同一条件のデータを検索
    updated = False
    for entry in data:
        if (entry["algorithm_name"] == "Dinamic Problem" and
            entry["vertex_number"] == vertex_number and
            entry["file_id"] == file_id and
            entry["t_max"] == t_max and
            entry["member"] == member and
            entry["probability"] == probability):
            # 条件が一致する場合はデータを上書き
            entry.update(new_entry)
            updated = True
            break
    
    # 条件が一致しない場合は新規追加
    if not updated:
        data.append(new_entry)
    
    # JSONファイルに保存
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    vertex_number_list = [4, 7, 10, 15, 21, 32, 33, 64, 66, 100]
    file_id_list = ["a", "b", "c", "d", "e"]
    t_max = 150
    member = 2
    probability = 0.8
    for vertex_number in vertex_number_list:
        for file_id in file_id_list:
            start = time.time()
            dinamic_problem = TOP_DP(file_id, vertex_number, t_max, member, probability)
            opt_value, opt_paths = dinamic_problem.dp_main()  # 出力: 40
            end = time.time()
            exe_time = end - start
            write_json(opt_value, opt_paths, exe_time)
            print(vertex_number, file_id, opt_value)
