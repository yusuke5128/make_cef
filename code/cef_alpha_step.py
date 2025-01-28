import json
import math
import os
import re
from typing import List, Tuple

import matplotlib.pyplot as plt
import scipy.stats as stats
from ellipse_solver import SolverEllipseMethod
from scipy.stats import norm


# vertexは単数形、verticesは複数形
# 自分が作った定式化。「一定確率で得られる報酬の最大化」0からスタートしてn+1に戻る。
class ConvexEllipse:
    def __init__(self, vertex_number: int, file_id: str, member: int, t_max_ratio: int, probability: float, alpha_how_many: float, outer_loop: int, inner_loop: int)  -> None:
        """
        楕円法のヒューリスティックで擬似的な凸包を作成して近似解を求めるクラス
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
        self.vertex_num = int(vertex_number)
        self.vertex_num_hikisu = vertex_number
        self.file_id = file_id
        self.vertices, self.vertices_distances, self.scores_ave, self.scores_var, self.all_nodes_cost = self.load_json_data(vertex_number)
        self.vertex_num = int(self.vertex_num)
        self.standard_deviation = [math.sqrt(var) for var in self.scores_var]
        self.t_max = t_max_ratio * self.all_nodes_cost
        self.member = member
        self.probability = probability
        self.convex_efficient_frontier = []
        self.x_cef = []
        self.c_p = -1 * stats.norm.ppf(probability)
        self.alalpha_how_many = alpha_how_many
        self.obj_value = []
        self.output_dir = f"proposed_method/vertex_num_{self.vertex_num}"
        self.outer_loop = outer_loop
        self.inner_loop = inner_loop


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
        solver_ellipse = SolverEllipseMethod(self.vertex_num_hikisu, self.file_id, self.member, self.t_max, self.probability, self.outer_loop, self.inner_loop)
        m_2, v_2, edge_route = solver_ellipse.main(alpha=1)
        self.convex_efficient_frontier.append([m_2, v_2])   # 第二初期解の平均値と分散を格納
        self.x_cef.append(edge_route)   # 第二初期解のルートを格納
        print("--------初期解の生成が終了しました。-------\n")
        print("初期解はm1:", m_1, ",v1:", v_1, ",m2:", m_2, ",v_2:", v_2, "です。")

        # step2: CEF（パレート最適解）
        self.find_efficient_point(m_1, v_1, m_2, v_2)   # 暫定解はここで格納

        # step3: CEFから厳密解
        optimal_index = self.find_optimum_index()
        opt_value = self.obj_value[optimal_index - 1]
        # opt_value, opt_path = self.make_output_data(optimal_index)
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
        for alpha in [i / self.alalpha_how_many for i in range(0, self.alalpha_how_many + 1)]:
            solver_ellipse = SolverEllipseMethod(self.vertex_num_hikisu, self.file_id, self.member, self.t_max, self.probability, self.outer_loop, self.inner_loop)
            m_new, v_new, edge_route = solver_ellipse.main(alpha)
            self.convex_efficient_frontier.append([m_new, v_new])
            self.x_cef.append(edge_route)


    def formula_alpha(self, m_1: float, v_1: float, m_2: float, v_2: float)   -> float:
        if (m_2 - m_1) + (v_2 - v_1) == 0:
            return 0
        alpha = (v_2 -v_1) / ((m_2 - m_1) + (v_2 - v_1))
        return alpha
    

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
        self.print_terminal(optimal_index)
        self.make_json(optimal_index)
        opt_value, opt_path = self.make_txt(optimal_index)
        return opt_value, opt_path

    def print_terminal(self, optimal_index: int) -> None:
        
        # 初期メッセージを表示
        print("\n\n==== 暫定解の生成結果 ====")
        print("暫定解 0 のとき経路はありません。m_y, v_y は", self.convex_efficient_frontier[0])
        
        # 暫定解を順に表示
        for index in range(len(self.x_cef)):
            print(f"\n---- 暫定解 {index + 1} ----")
            print(f"目的関数値: {self.obj_value[index]}")
            print(f"[m_y, v_y] = {self.convex_efficient_frontier[index + 1]}")
            
            # グラフを描画
            # self.result_draw_graph(index, str(index + 1))
        
        # 最適解の表示
        print("\n==== 最適解 ====")
        print(f"最適解のインデックス: {optimal_index}")
        print(f"最適値: {self.obj_value[optimal_index - 1]}")
        print(f"[m_y, v_y]: {self.convex_efficient_frontier[optimal_index]}")
        
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
            paths = self.extract_paths(index)
            output_lines.append("経路:")
            output_lines.extend([str(path) for path in paths])

        # 最適解の表示
        output_lines.append("\n==== 最適解 ====")
        output_lines.append(f"最適解のインデックス: {optimal_index}")
        output_lines.append(f"最適値: {self.obj_value[optimal_index - 1]}")
        output_lines.append(f"[m_y, v_y]: {self.convex_efficient_frontier[optimal_index]}")
        
        # 最適解の経路を各要素ごとに改行
        optimal_path = self.extract_paths(optimal_index - 1)
        output_lines.append("最適解の経路:")
        output_lines.extend([str(path) for path in optimal_path])
        
        # ファイルパスの設定
        txt_path = os.path.join(self.output_dir, "txt", f"{self.vertex_num}_{self.file_id}.txt")

        # ファイルに保存
        with open(txt_path, "w") as f:
            for line in output_lines:
                f.write(line + "\n")
                
        return self.obj_value[optimal_index - 1], self.sub_extract_paths(optimal_index - 1)


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
        self.output_dir = f"proposed_method/trash/vertex_num_{self.vertex_num}"
        
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
    def load_json_data(self, vertex_num: str)   -> Tuple[List[List[int]], List[List[int]], List[int], int]:
        dataset_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../previous_dataset')
        print(vertex_num)
        json_files = [f for f in os.listdir(dataset_directory) if f.endswith('.json')]
        pattern = fr'vertex_{vertex_num}_{self.file_id}\.json$'
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
        return None, None, None, None, None


