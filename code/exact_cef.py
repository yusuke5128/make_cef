import json
import math
import os
import re
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx
import scipy.stats as stats
from dp import TOP_DP
from pyscipopt import Model, quicksum
from scipy.stats import norm


# vertexは単数形、verticesは複数形
# 自分が作った定式化。「一定確率で得られる報酬の最大化」0からスタートしてn+1に戻る。
class CEF_DP:
    def __init__(self, file_id: str, vertex_number: int, t_max_ratio: int, member: int, probability: float)  -> None:
        """
        self.vertex_num: 頂点の個数（スタート、ゴール含む）
        self.vertives: 各頂点の二次元座標
        self.vertices_distances: 各点間の距離(二次元リスト)
        self.scores_ave: 各点のスコアの平均（一次元リスト）
        self.scores_var: 各点のスコアの分散（一次元リスト）
        self.target_reward: 目標報酬
        self.standard_deviation: 各頂点のスコアの標準偏差
        self.t_max: T_{max}
        self.member: チーム数
        self.probbility: 成功確率
        self.convex_efficient_frontier: 暫定解のm,vを格納する並列
        self.x_cef: 暫定解のx_ij行列を格納する
        self.c_p: 確率probabilityの時の偏差
        self.obj_value: 目的関数値を格納
        self.output_dir: 作成したグラフを保存するディレクトリを指定
        """
        # jsonファイルからデータを受け取る
        self.vertex_num = vertex_number
        self.file_id = file_id
        self.vertices, self.vertices_distances, self.scores_ave, self.scores_var, self.all_nodes_cost = self.load_json_data()
        self.standard_deviation = [math.sqrt(var) for var in self.scores_var]
        self.t_max = t_max_ratio * self.all_nodes_cost
        self.member = member
        self.probability = probability
        self.convex_efficient_frontier = []
        self.x_cef = []
        self.c_p = -1 * stats.norm.ppf(probability)
        self.obj_value = []
        self.output_dir = f"proposed_method/vertex_num_{self.vertex_num}"


    def main_exact_algorithm(self)  -> Tuple[List, float]:
        """main関数

        Returns:
            Tuple[List, float]: 最適解のパスと目的関数値を出力
        """
        # 前回のファイルを削除
        self.setup_directories()
        
        # step1:初期解
        m_1, v_1 = 0, 0 # TODO 確認。ある値x以上になる確率がprobabilityとなるxをm_1(初期解の平均値)とした。
        self.convex_efficient_frontier.append([m_1, v_1])

        top_dp = TOP_DP(self.file_id, self.vertex_num, self.t_max, self.member, self.probability, alpha=1)
        m_2, v_2, edge_route = top_dp.dp_main()
        # m_2, v_2, edge_route = self.sol_by_scip_qa_alpha(alpha=1)
        self.convex_efficient_frontier.append([m_2, v_2])   # 第二初期解の平均値と分散を格納
        self.x_cef.append(edge_route)   # 第二初期解のルートを格納
        print("--------初期解の生成が終了しました。-------\n")
        print("初期解はm1:", m_1, ",v1:", v_1, ",m2:", m_2, ",v_2:", v_2, "です。")

        # step2: CEF（パレート最適解）
        self.find_efficient_point(m_1, v_1, m_2, v_2)   # 暫定解はここで格納

        # step3: CEFから厳密解
        optimal_index = self.find_optimum_index()
        opt_value = self.make_output_data(optimal_index)
        print(f"{self.convex_efficient_frontier=}")
        opt_path = [[0, 1],[1,2]]
        return opt_value, self.convex_efficient_frontier
        

    def find_optimum_index(self)    -> int:
        # 最適なインデックスと最大値を初期化
        optimal_index = -1
        max_value = float('-inf')
        sub_convex_efficient_frontier = self.convex_efficient_frontier[1:]   # 第一初期解は省く
        print(sub_convex_efficient_frontier)
        # convex efficient frontier の (m, v) ペアをすべてループ
        for index, (m, v) in enumerate(sub_convex_efficient_frontier):
            value = m + self.c_p * math.sqrt(v)
            self.obj_value.append(value)
            print("max_valueの変遷", max_value)
            # これまでの最大値を超えるかどうかを確認
            if value > max_value:
                optimal_index = index+1 # 第一初期解分の帳尻合わせ
                max_value = value
        return optimal_index

    def find_efficient_point(self, m_1: float, v_1: float, m_2: float, v_2: float)    -> None:
        # step1
        alpha = self.formula_alpha(m_1, v_1, m_2, v_2)
        print("アルファ", alpha)
        top_dp = TOP_DP(self.file_id, self.vertex_num, self.t_max, self.member, self.probability, alpha)
        m_new, v_new, edge_route = top_dp.dp_main()
        # m_new, v_new, edge_route = self.sol_by_scip_qa_alpha(alpha)
        # print(self.convex_efficient_frontier)
        if [m_new, v_new] not in self.convex_efficient_frontier:
            # step2.1
            self.convex_efficient_frontier.append([m_new, v_new])
            self.x_cef.append(edge_route)
            # step2.2
            self.find_efficient_point(m_new, v_new, m_2, v_2)
            # step2.3
            self.find_efficient_point(m_1, v_1, m_new, v_new)
        else:
            return


    def formula_alpha(self, m_1: float, v_1: float, m_2: float, v_2: float)   -> float:
        if ((m_2 - m_1) + (v_2 - v_1)) != 0:
            alpha = (v_2 -v_1) / ((m_2 - m_1) + (v_2 - v_1))
        else:
            alpha = 0
        return alpha
    
    def sol_by_scip_qa_alpha(self, alpha: float)   -> Tuple[int, int, List[List[int]]]:
        """
        alpha: 定数alpha、初期解の時は常に1
        """

        # 変数の定義
        model = Model("team_orienteering")

        # x_mijの定義を修正
        x_mij = [[[model.addVar(f"x_{m}_{i}_{j}", vtype="B") for j in range(self.vertex_num)] for i in range(self.vertex_num)] for m in range(self.member)]
        y_i = [model.addVar(f"y_{i}", vtype="B") for i in range(self.vertex_num)]
        
        # 目的関数の記述
        objective = alpha * sum(self.scores_ave[i] * y_i[i] for i in range(self.vertex_num))
        if alpha != 1:
            objective -= (1 - alpha) * sum(self.scores_var[i] * y_i[i] for i in range(self.vertex_num))
        model.setObjective(objective, "maximize")
        # 目的関数の記述終了
        
        E = [[1 if i != j else 0 for j in range(self.vertex_num)] for i in range(self.vertex_num)]  # エッジ行列を作成
        
        # ここからは部分巡回路除去制約を実装する。肝である。
        # "https://github.com/scipopt/PySCIPOpt/blob/master/examples/finished/tsp.py"を参考にした。
        def addcut(one_path):
            """subtour_elimination
            求め方は、各メンバーのパスについて連結成分の個数を取得。その個数が1ではなかった場合に、部分巡回路があるとした。
            そして、見つけた部分巡回路を制約として追加。

            Args:
                eone_path: 通る辺を接続成分としたリスト（メンバー1人分を格納）
            """
            G = networkx.Graph()
            G.add_edges_from(one_path)
            Components = list(networkx.connected_components(G))
            if len(Components) == 1:
                return False
            model.freeTransform()
            for S in Components:
                model.addCons(quicksum(x_mij[m][i][j] for i in S for j in S for m in range(self.member)) <= len(S) - 1)
            return True
        
        # TODO あるメンバーがノードに訪れた場合に他のメンバーがそのノードを訪れないようにする制約（山口先生に確認）
        # 訪問制約を追加（同じノードに他のメンバーが訪れないようにする）
        """
        for i in range(1, self.vertex_num):  # スタートを除く
            model.addCons(quicksum(x_mij[m][i][j] for j in range(self.vertex_num) for m in range(self.member)) <= 1, f"visit_once_{i}")
        
        # TODO 上のTODOで追加した制約だと、スタート→ゴールのみの経路が複数存在する場合を除去できていなかったので、1度しかそれをできないようにした。
        model.addCons(quicksum(x_mij[m][0][self.vertex_num - 1] for m in range(self.member)) <= 1, "start_goal")
        """
        #制約条件1
        for i in range(self.vertex_num):
            const_expr = quicksum(x_mij[m][i][j] for j in range(self.vertex_num) for m in range(self.member))
            model.addCons(y_i[i] - const_expr <= 0, f"constraint_y_{i}")
            
        # 制約条件2
        for m in range(self.member):
            for i in range(1, self.vertex_num - 1):  # 0とn（最終ノード）は除外
                flow_in_expr = quicksum(x_mij[m][j][i] for j in range(self.vertex_num) if E[j][i] == 1)
                flow_out_expr = quicksum(x_mij[m][i][j] for j in range(self.vertex_num) if E[i][j] == 1)
                model.addCons(flow_in_expr - flow_out_expr == 0, f"flow_conservation_{m}_{i}")
                
        # 制約条件3
        for m in range(self.member):
            model.addCons(
                quicksum(x_mij[m][0][j] for j in range(self.vertex_num)) == 1,
                f"constraint_start_{m}"
            )

        # 制約条件4
        n_goal_node = self.vertex_num - 1
        for m in range(self.member):
            model.addCons(
                quicksum(x_mij[m][i][n_goal_node] for i in range(self.vertex_num)) == 1,
                f"constraint_goal_{m}"
            )

        # 制約条件5
        for m in range(self.member):
            model.addCons(
                quicksum(self.vertices_distances[i][j] * x_mij[m][i][j] for i in range(self.vertex_num) for j in range(self.vertex_num) if E[i][j]==1) <= self.t_max,
                f"distance_constraint_{m}"
            )
            
        """    
        # 制約条件6（部分巡回路除去制約）
        for m in range(self.member):
            for k in range(1, self.vertex_num):
                subsets = combinations(range(1, self.vertex_num), k)  # 頂点のインデックスを使う
                for V in subsets:
                    # エッジのリストを x_ij を用いて構築
                    edges_in_V = [x_mij[m][i][j] for i in V for j in V]

                    # 制約の追加: Σ x_ij <= |V| - 1
                    if edges_in_V:
                        model.addCons(
                            sum(edges_in_V) <= len(V) - 1
                        )
        """
        isMIP = False
        while True: # 部分巡回路が除去できていない間続ける。
            # 問題を解く
            try:
                model.optimize()
                # 解のステータスを確認
                status = model.getStatus()
                print(f"解のステータス: {status}")

                if status != "optimal":
                    raise Exception("Optimal solution is not feasible")
                # 解のステータスを確認
                if model.getStatus() != "optimal":
                    raise Exception("Optimal solution is not feasible")

                # 解の取得
                edge_route = [[[1 if model.getVal(x_mij[m][i][j]) >= 0.5 else 0 for j in range(self.vertex_num)] for i in range(self.vertex_num)] for m in range(self.member)]                        
                all_members_corrected = True  # 全メンバーが単一の巡回路かどうかを判定するフラグ（Trueで初期化）

                # 各メンバーごとに部分巡回路除去カットを追加
                for m in range(self.member):
                    cut_edges = []
                    for i in range(self.vertex_num):
                        for j in range(self.vertex_num):
                            if edge_route[m][i][j] == 1:
                                cut_edges.append((i, j))
                    
                    # 制約条件6: 部分巡回路除去制約
                    if addcut(cut_edges) == False:
                        continue  # このメンバーは単一の巡回路であるため次のメンバーへ
                    else:
                        all_members_corrected = False  # 部分巡回路があるのでFalseにする
                                
                # すべてのメンバーが単一の巡回路でつながっている場合
                if all_members_corrected:
                    y_values = [1 if model.getVal(y_i[i]) >= 0.5 else 0 for i in range(self.vertex_num)]

                    # ルートの平均スコアと分散スコアを計算
                    route_sum_ave = sum(y * ave for y, ave in zip(y_values, self.scores_ave))
                    route_sum_var = sum(y * var for y, var in zip(y_values, self.scores_var))

                    return route_sum_ave, route_sum_var, edge_route

            except Exception as e:
                print(f"SCIP Solver Error: {e}")
                raise
            
    def init_ave(self)  -> float:
        sum_ave = sum(self.scores_ave)
        sum_var = math.sqrt(sum(self.scores_var))
        x = sum_ave + sum_var * norm.ppf(1 - self.probability)
        print("init_ave", x)
        return x
        
    def result_draw_graph(self, route_index: int, mark: str)  -> None:
        edge_route = self.x_cef[route_index]
        plt.figure(figsize=(8, 8))
        # 頂点をプロット
        for i, (x, y) in enumerate(self.vertices):
            plt.scatter(x, y, color='blue', s=40)
            plt.text(x, y, f"{i}", fontsize=10, ha='right')

        # 自動的に色を生成する
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']

        # 辺をプロット
        for m in range(self.member):
            for i in range(self.vertex_num):
                for j in range(self.vertex_num):
                    if edge_route[m][i][j] > 0:  # 通った辺だけを描画
                        x_values = [self.vertices[i][0], self.vertices[j][0]]
                        y_values = [self.vertices[i][1], self.vertices[j][1]]
                        plt.plot(x_values, y_values, color=colors[m % len(colors)], linewidth=2)

        plt.title("Optimal Route")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True)
        graph_path = os.path.join(self.output_dir, "graph_plot")
        file_name = f"{graph_path}/{self.vertex_num}_{self.file_id}_{mark}.png"
        plt.savefig(file_name)  # グラフをファイルに保存
        plt.close()  # プロットを閉じる
        
    def extract_paths(self, route_index: int) -> None:
        """
        与えられた3次元配列から、各メンバーの通過したノードを→で繋いだ経路を表示する関数。

        Args:
            route_index (int): どのパスを表示したいかを指定

        Returns:
            paths (list): 各メンバーの経路を→で繋いだ形式で格納したリスト。
        """
        paths = []
        paths_plain = []
        route_data = self.x_cef[route_index]
        for m in range(self.member):
            path = []
            current_node = 0  # 各メンバーは0からスタートと仮定
            visited = set()  # 訪問済みのノードを追跡するためのセット

            while current_node not in visited:  # 無限ループ防止
                path.append(current_node)
                visited.add(current_node)
                found_next = False

                for j in range(self.vertex_num):
                    if route_data[m][current_node][j] == 1:
                        current_node = j
                        found_next = True
                        break
                
                if not found_next:
                    break  # 経路の終点に到達

            paths.append(" → ".join(map(str, path)))
            paths_plain.append(path)
        return paths, paths_plain
    
    def sub_extract_paths(self, route_index: int) -> None:
        """
        与えられた3次元配列から、各メンバーの通過したノードを→なしで繋いだ経路を表示する関数。

        Args:
            route_index (int): どのパスを表示したいかを指定

        Returns:
            paths (list): 各メンバーの経路を→で繋いだ形式で格納したリスト。
        """
        paths_plain = []
        route_data = self.x_cef[route_index]
        for m in range(self.member):
            path = []
            current_node = 0  # 各メンバーは0からスタートと仮定
            visited = set()  # 訪問済みのノードを追跡するためのセット

            while current_node not in visited:  # 無限ループ防止
                path.append(current_node)
                visited.add(current_node)
                found_next = False

                for j in range(self.vertex_num):
                    if route_data[m][current_node][j] == 1:
                        current_node = j
                        found_next = True
                        break
                
                if not found_next:
                    break  # 経路の終点に到達
            paths_plain.append(path)
        return paths_plain

            
    def make_output_data(self, optimal_index: int)  -> Tuple[List, float]:
        # self.print_terminal(optimal_index)
        # self.make_json(optimal_index)
        opt_value = self.make_txt(optimal_index)
        return opt_value

    def print_terminal(self, optimal_index: int) -> None:
        
        # 初期メッセージを表示
        print("\n\n==== 暫定解の生成結果 ====")
        print("暫定解 0 のとき経路はありません。m_y, v_y は", self.convex_efficient_frontier[0])
        
        # 暫定解を順に表示
        for index in range(len(self.x_cef)):
            print(f"\n---- 暫定解 {index + 1} ----")
            print(f"目的関数値: {self.obj_value[index]}")
            print(f"[m_y, v_y] = {self.convex_efficient_frontier[index + 1]}")
            
            for path in self.extract_paths(index):
                print(path)
            # グラフを描画
            self.result_draw_graph(index, str(index + 1))
        
        # 最適解の表示
        print("\n==== 最適解 ====")
        print(f"最適解のインデックス: {optimal_index}")
        print(f"最適値: {self.obj_value[optimal_index - 1]}")
        print(f"[m_y, v_y]: {self.convex_efficient_frontier[optimal_index]}")
        
        # 最適解の経路を出力し、最適解グラフを描画
        print("各メンバーが通ったパスは")
        for path in self.extract_paths(optimal_index - 1):
            print(path)
        self.result_draw_graph(optimal_index - 1, "opt")
        
    def make_txt(self, optimal_index: int) -> Tuple[List, float]:
        # 出力内容を保持するリスト
        output_lines = []

        # 初期メッセージ
        output_lines.append("\n\n==== 暫定解の生成結果 ====")
        output_lines.append(f"暫定解 0 のとき経路はありません。m_y, v_y は {self.convex_efficient_frontier[0]}")
        
        # 暫定解を順に追加
        for index in range(len(self.x_cef)):
            output_lines.append(f"\n---- 暫定解 {index + 1} ----")
            output_lines.append(f"目的関数値: {self.obj_value[index]}")
            output_lines.append(f"[m_y, v_y] = {self.convex_efficient_frontier[index + 1]}")
            
            # 経路情報（出力内容に含める場合）を各要素ごとに改行
            # paths = self.extract_paths(index)
            # output_lines.append("経路:")
            # output_lines.extend([str(path) for path in paths])

        # 最適解の表示
        output_lines.append("\n==== 最適解 ====")
        output_lines.append(f"最適解のインデックス: {optimal_index}")
        output_lines.append(f"最適値: {self.obj_value[optimal_index - 1]}")
        output_lines.append(f"[m_y, v_y]: {self.convex_efficient_frontier[optimal_index]}")
        
        # 最適解の経路を各要素ごとに改行
        # optimal_path = self.extract_paths(optimal_index - 1)
        # output_lines.append("最適解の経路:")
        # output_lines.extend([str(path) for path in optimal_path])
        
        # ファイルパスの設定
        txt_path = os.path.join(self.output_dir, "txt", f"{self.vertex_num}_{self.file_id}.txt")

        # ファイルに保存
        with open(txt_path, "w") as f:
            for line in output_lines:
                f.write(line + "\n")
                
        # return self.obj_value[optimal_index - 1], self.sub_extract_paths(optimal_index - 1)
        return self.obj_value[optimal_index - 1]


    def make_json(self, optimal_index: int) -> None:
        # データ構造を作成
        data = {
            "暫定解": [
                {
                    "解": 0,
                    "目的関数値": "なし",
                    "m_y": self.convex_efficient_frontier[0][0],
                    "v_y": self.convex_efficient_frontier[0][1]
                }
            ]
        }

        # 暫定解を順に追加
        for index in range(len(self.x_cef)):
            paths = self.extract_paths(index)  # 経路情報
            data["暫定解"].append({
                "解": index + 1,
                "目的関数値": self.obj_value[index],
                "m_y": self.convex_efficient_frontier[index + 1][0],
                "v_y": self.convex_efficient_frontier[index + 1][1],
                "経路": paths
            })

        # 最適解の情報
        optimal_path = self.extract_paths(optimal_index - 1)
        data["最適解"] = {
            "解": optimal_index,
            "目的関数値": self.obj_value[optimal_index - 1],
            "m_y": self.convex_efficient_frontier[optimal_index][0],
            "v_y": self.convex_efficient_frontier[optimal_index][1],
            "経路": optimal_path
        }
        
        # 出力先ディレクトリとファイルパスを設定
        json_path = os.path.join(self.output_dir, "json")
        file_path = os.path.join(json_path, f"{self.vertex_num}_{self.file_id}.json")

        # JSONファイルに保存
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def setup_directories(self) -> None:
        self.output_dir = f"proposed_method/vertex_num_{self.vertex_num}"
        
        # 新しいディレクトリを作成し、サブディレクトリも作成（存在しない場合のみ）
        graph_path = os.path.join(self.output_dir, "graph_plot")
        os.makedirs(graph_path, exist_ok=True)
        self.remove_file(graph_path)
        
        txt_path = os.path.join(self.output_dir, "txt")
        os.makedirs(txt_path, exist_ok=True)
        self.remove_file(txt_path)
        
        json_path = os.path.join(self.output_dir, "json")
        os.makedirs(json_path, exist_ok=True)
        self.remove_file(json_path)

    def remove_file(self, directory: str)   -> None:
        pattern = re.compile(rf"{self.vertex_num}_{self.file_id}")
        print("pattern", pattern)
        # フォルダ内のファイルをループし、パターンに一致するファイルを削除
        for filename in os.listdir(directory):
            if pattern.match(filename):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
    
    # jsonからデータを取ってくる関数
    def load_json_data(self)   -> Tuple[List[List[int]], List[List[int]], List[int], int]:
        dataset_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../previous_dataset')
        json_files = [f for f in os.listdir(dataset_directory) if f.endswith('.json')]
        pattern = fr'vertex_{self.vertex_num}_{self.file_id}\.json$'
        # 指定した数字がファイル名に含まれているか確認
        for file_name in json_files:
            if re.search(pattern, file_name):
                json_path = os.path.join(dataset_directory, file_name)
                # JSONファイルを読み込む
                with open(json_path, 'r') as json_file:
                    data = json.load(json_file)
                    self.vertives = data.get('vertices', [])
                    self.vertices_distances = data.get('distances', [])
                    self.scores_ave = data.get('scores_ave', [])
                    self.scores_var = data.get('scores_var', [])
                    self.all_nodes_cost = int(data.get("minimum_distances", {}).get("cost", 0))
                
                print(f"ファイル {file_name} の中身を読み込みました。")
                return self.vertives, self.vertices_distances, self.scores_ave, self.scores_var, self.all_nodes_cost
        print(f"{self.vertex_num}を含むJSONファイルは見つかりませんでした。")
        return None, None, None, None
