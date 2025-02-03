import copy
import heapq
import json
import os
import random
import re
import sys
from typing import List, Tuple

from scipy import stats


class SolverEllipseMethod:
    def __init__(self, vertex_number: str, file_id: str, member: int, t_max: float, probability: float, outer_loop: int, inner_loop: int):
        """楕円法において非最適パスは考慮しない提案手法

        Args:
            vertex_number (str): _description_
            file_id (str): _description_
            member (int): _description_
            t_max_ratio (float): _description_
            probability (float): _description_
        """
        self.vertex_num = int(vertex_number)
        self.start = 0
        self.goal = self.vertex_num - 1
        self.member = member
        self.deviation = 0 # どれくらい許容するか
        self.deviation_threshold = 0.05
        self.file_id = file_id
        csv_data = self.get_data_from_csv(vertex_number, file_id)
        self.distances = csv_data["distances"]
        self.scores_ave = csv_data["scores_ave"]
        self.scores_var = csv_data["scores_var"]
        self.t_max = t_max
        self.nodes_in_ellipse = self.get_ellipse_nodes()
        self.unused_nodes = []
        self.c_p = -1 * stats.norm.ppf(probability)
        self.record = 0
        self.outer_loop = outer_loop
        self.inner_loop = inner_loop
        self.score_all_change = []
        self.alpha = 0
        
    def main(self, alpha) -> List[List[int]]:
        """メイン処理

        Returns:
            List[List[int]]: 最適解を導くパス
        """
        self.alpha = alpha
        
        team_paths = self.create_initial_solution()
        print("初期解生成完了")
        print(f"{self.calculate_team_score(team_paths)=}")
        print(f"{team_paths=}")
        self.debug_main(team_paths)
        
        # 1回目の最適化
        team_score = self.calculate_team_score(team_paths)
        self.record = team_score
        self.deviation = self.deviation_threshold * self.record
        team_paths = self.improvement(team_paths, "inner")
        
        print("========================================================================1回目の最適化完了")
        self.debug_main(team_paths)
        
        remove_count = 2
        print("Reinitialize 2開始")
        print(f"{self.calculate_team_score(team_paths)=}")
        print(f"{team_paths=}")
        team_paths = self.reinitialization_two(remove_count, team_paths)
        print("Reinitialize 2完了")
        print(f"{self.calculate_team_score(team_paths)=}")
        print(f"{team_paths=}")
        self.debug_main(team_paths)
        
        # 2回目の最適化
        self.deviation_threshold = 0.025
        team_paths_score = self.calculate_team_score(team_paths)
        record = team_paths_score
        self.deviation = self.deviation_threshold * record
        team_paths = self.improvement(team_paths, "outer")
        print("2回目の最適化完了")
        print(f"{self.calculate_team_score(team_paths)=}")
        print(f"{team_paths=}")
        self.debug_main(team_paths)
        
        ave, var = 0, 0
        for path in team_paths:
            for node in path:
                ave += self.scores_ave[node]
                var += self.scores_var[node]
        print(f"{self.alpha=}, {ave=}, {var=}")
        return ave, var, team_paths
        
        
    def reinitialization_two(self, remove_node_count: int, team_paths: List[List[int]]) -> List[List[int]]:
        """Reinitialization 2を行う．

        Args:
            remove_node_count (int): 削除するノードの数
            paths_opt (List[List[int]]): チームパス

        Returns:
            List[List[int]]: ノードを削除した後の最適解のパスのリスト
        """
        for path_index, path in enumerate(team_paths):
            remove_node_count = int(self.get_adjusted_path_length(path) / 4)
            path_ave = sum(self.scores_ave[node] for node in path)
            path_var = sum(self.scores_var[node] for node in path)
            path_score = self.calculate_obj_function(path_ave, path_var)
            for _ in range(remove_node_count):
                min_ratio = float("inf")
                best_index = -1
                for node_index in range(1, len(path) - 1):  # ここは少し効率が悪いかも．けど，高々remove_node_countが3程度なので問題ないとする．
                    if path[node_index] == self.start or path[node_index] == self.goal:
                        continue
                    cost_change = self.remove_cost_change(node_index, path)
                    path_ave_after = path_ave - self.scores_ave[path[node_index]]
                    path_var_after = path_var - self.scores_var[path[node_index]]
                    path_score_change = path_score - self.calculate_obj_function(path_ave_after, path_var_after)
                    change_ratio = path_score_change / cost_change
                    if change_ratio < min_ratio:
                        # パスコストがt_maxを超えない場合のみ削除したい．
                        if self.calculate_path_cost(path) + cost_change <= self.t_max:
                            min_ratio = change_ratio
                            best_index = node_index
                if best_index != -1:
                    remove_node = path.pop(best_index)
                    self.unused_nodes.append(remove_node)
            team_paths[path_index] = path
        return team_paths
        
    def get_adjusted_path_length(self, path):
        if path and path[0] == self.start:
            path = path[1:]  # 最初の要素を除く
        if path and path[-1] == self.goal:
            path = path[:-1]  # 最後の要素を除く
        return len(path)
    
    def improvement(self, team_paths: List[List[int]], loop: str) -> List[List[int]]:
        """
        初期解を改善していく処理
        Args:
            team_paths (List[List[int]]): チームのパス
        Returns:
            List[List[int]]: 改善後のチームのパス
        """
        # TODO このループ処理が本当に正しいか確認する
        not_change_outer_loop_count = 0  # 外側ループで変化がなかった回数
        before_team_score = 0
        for k in range(self.outer_loop):
            for _ in range(self.inner_loop):
                check_change_flag_inner_loop = False
                team_paths_before = copy.deepcopy(team_paths)
                print("2-point exchange開始")
                print(f"{self.calculate_team_score(team_paths)=}")
                print(f"{team_paths=}")
                self.debug_main(team_paths)
                team_paths = self.two_point_exchange(team_paths)
                print("2-point exchange完了")
                print(f"{self.calculate_team_score(team_paths)=}")
                print(f"{team_paths=}")
                self.debug_main(team_paths)
                print("one-point move開始")
                print(f"{self.calculate_team_score(team_paths)=}")
                print(f"{team_paths=}")
                self.one_point_movement(team_paths)
                print("one-point move完了")
                print(f"{self.calculate_team_score(team_paths)=}")
                print(f"{team_paths=}")
                self.debug_main(team_paths)
                print("clean up開始")
                print(f"{self.calculate_team_score(team_paths)=}")
                print(f"{team_paths=}")
                self.clean_up(team_paths)
                print("clean up完了")
                print(f"{self.calculate_team_score(team_paths)=}")
                print(f"{team_paths=}")
                # print(f"{team_paths=}")
                self.debug_main(team_paths)
                
                check_change_flag_inner_loop = self.check_change_or_not(team_paths_before, team_paths)
                if check_change_flag_inner_loop is False:
                    break
                else:
                    before_team_score = self.calculate_team_score(team_paths_before)
                    now_team_score = self.calculate_team_score(team_paths)
                    if now_team_score > before_team_score:
                        self.record = now_team_score
                        self.deviation = self.deviation_threshold * self.record
                    
            if self.record == before_team_score:
                not_change_outer_loop_count += 1
            else:
                not_change_outer_loop_count = 0
                
            if not_change_outer_loop_count == 5:    # 外側ループで5回以上変化がなかった場合
                break
            """
            if loop == "inner":
                print("Reinitialize 1開始")
                self.debug_main(team_paths)
                self.reinitialization_one(k, team_paths)
                print("Reinitialize 1完了")
                self.debug_main(team_paths)
            """
        return team_paths
    
    def reinitialization_one(self, k: int, team_paths: List[List[int]]) -> List[List[int]]:
        """Reinitialization 1を行う
        
        Args:
            k (int): paths_optの点から削除するノードの数
            team_paths (List[List[int]]): チームのパス
            
        Returns:
            List[List[int]]: 処理後のチームのパス
        """
        node_diffs = []
        k = int(k / 20) + 1
        for path in team_paths:
            ave, var = sum(self.scores_ave[node] for node in path),  sum(self.scores_var[node] for node in path)
            path_cost = self.calculate_path_cost(path)
            path_score = self.calculate_obj_function(ave, var) # 戻ってこい
            for node_in_path_index, node_in_path in enumerate(path):
                if node_in_path == self.start or node_in_path == self.goal:
                    continue
                remove_cost = self.remove_cost_change(node_in_path_index, path)
                new_opt_path_cost = path_cost + remove_cost
                if new_opt_path_cost > self.t_max:
                    continue
                ave_tmp, var_tmp = ave - self.scores_ave[node_in_path], var - self.scores_var[node_in_path]
                path_score_tmp = self.calculate_obj_function(ave_tmp, var_tmp)
                diff = path_score - path_score_tmp
                # ヒープ用にタプル(-diff, node)を追加（最小ヒープで最大値を扱う場合は負符号を使用）
                heapq.heappush(node_diffs, (diff, node_in_path))    # TODO ここで本当に最小のdiffが取得できているか確認
        # diffが小さいk個を取得（ヒープからpop）
        smallest_k_nodes = [heapq.heappop(node_diffs)[1] for _ in range(min(k, len(node_diffs)))]
        print(f"{smallest_k_nodes=}, {team_paths=}")
        for path_index, path in enumerate(team_paths):
            for node in smallest_k_nodes:
                if node in path:
                    self.unused_nodes.append(node)
                    path.remove(node)
                    team_paths[path_index] = path
        print(f"{team_paths=}")
        return team_paths
    
    def clean_up(self, team_paths: List[List[int]]) -> List[List[int]]:
        """clean upの処理全般

        Args:
            paths_opt (List[int]): チーム（最適パス）
            paths_no_opt (List[int]): 非最適パス

        Returns:
            List[int]: 改善後のチーム（最適パス）
        """
        for path_index, path in enumerate(team_paths):
            path = self.two_opt_algorithms(path)    # 2-optアルゴリズム
            team_paths[path_index] = path
        return team_paths
    
    def two_opt_algorithms(self, path: List[int])   -> List[int]:
        """各パスについてtwo optを行う

        Args:
            path (List[int]): パス（1つ）
            
        Returns:
            List[int]: 2-optを行った後のパス
        """
        while True:
            improved = False
            for i in range(1, len(path) - 2):
                for j in range(i + 1, len(path) - 1):   # パスの始点と終点はスワップ対象外
                    if self.two_opt_swap(path, i, j):
                        improved = True
            if not improved:
                break
        return path
    
    def two_opt_swap(self, path: List[int], i: int, j: int) -> bool:
        """2-optのスワップを行う

        Args:
            path (List[int]): パス（1つ）
            i (int): スワップするノードのインデックス
            j (int): スワップするノードのインデックス

        Returns:
            bool: パスが改善された場合はTrue、そうでない場合はFalse
        """
        A, B = path[i - 1], path[i]
        C, D = path[j], path[j + 1]
        if self.distances[A][B] + self.distances[C][D] > self.distances[A][C] + self.distances[B][D]:
            path[i:j + 1] = path[j:i - 1:-1]
            return True
        return False
                
    def one_point_movement(self, team_paths: List[List[int]]) -> List[List[int]]:
        """1点移動を行う
        Args:
            team_paths (List[List[int]]): チームのパス

        Returns:
            List[List[int]]: 1点移動後のチームのパス
        """
        nodes_loop = [node for node in self.nodes_in_ellipse if node not in (0, self.goal)]
        change_flag = False
        for node in nodes_loop: # 楕円内の点を1つずつ取り出す．
            best_node_index_sub = -1
            tmp_team_score = 0
            
            path_index, end_path_loop = 0, len(team_paths)
            while path_index < end_path_loop:
                best_path_index_sub = -1
                path = team_paths[path_index]
                path_cost = self.calculate_path_cost(path)  # TODO 要注意．path_costの計算場所がもう一つ後のwhileかも．
                path_ave, path_var = self.calculate_path_ave_var(path)
                path_score = self.calculate_obj_function(path_ave, path_var)
                if node in path:
                    path_index += 1
                    continue
                node_index, end_node_loop = 1, len(path)
                while node_index < end_node_loop:
                    sub_path_score, sub_path_new_score = 0, 0
                    path_add_cost_change = self.calculate_add_cost(path, node, node_index)
                    if path_cost + path_add_cost_change > self.t_max:
                        node_index += 1
                        continue
                    # もしもノードがチームパス内のノードにすでに含まれていたらそこから削除した場合を考える．
                    node_in_team_flag, node_in_team_path_index, node_in_team_node_index = self.check_node_in_team_yet(team_paths, node, path_index)
                    if node_in_team_flag:
                        sub_path_new_cost = self.calculate_remove_after_cost(team_paths[node_in_team_path_index], node_in_team_node_index)
                        if sub_path_new_cost > self.t_max:  # あるノードにすでに含まれていることでt_maxを超える場合→次のノードへ
                            node_index += 1
                            continue
                        sub_path_score = self.calculate_path_score(team_paths[node_in_team_path_index])
                        sub_path_new = copy.deepcopy(team_paths[node_in_team_path_index])
                        sub_path_new.pop(node_in_team_node_index)   # 新しいサブパスを格納
                        sub_path_new_score = self.calculate_path_score(sub_path_new)   # 新しいサブパスのスコア格納
                    path_ave_tmp, path_var_tmp = self.scores_ave[node] + path_ave, self.scores_var[node] + path_var
                    new_path_score = self.calculate_obj_function(path_ave_tmp, path_var_tmp)
                    now_team_score = self.calculate_team_score(team_paths)
                    new_team_score = now_team_score - path_score + new_path_score - sub_path_score + sub_path_new_score
                    if node_in_team_flag:
                        if now_team_score < new_team_score:
                            path.insert(node_index, node)
                            team_paths[path_index] = path
                            team_paths[node_in_team_path_index] = sub_path_new
                            change_flag = True
                            break
                        elif new_team_score > tmp_team_score:
                            tmp_team_score = new_team_score
                            best_path_index_sub = path_index
                            best_node_index_sub = node_index
                            sub_path_new_copy = copy.deepcopy(sub_path_new)
                    else:
                        # team_paths内にノードが含まれていない場合の処理を書く
                        if now_team_score < new_team_score:
                            path.insert(node_index, node)
                            team_paths[path_index] = path
                            break
                        elif new_team_score > tmp_team_score:
                            tmp_team_score = new_team_score
                            best_path_index_sub = path_index
                            best_node_index_sub = node_index
                            
                            
                    node_index += 1
                
                if change_flag is False and tmp_team_score > self.record - self.deviation and best_path_index_sub != -1:
                    path.insert(best_node_index_sub, node)
                    team_paths[best_path_index_sub] = path
                    if node_in_team_flag:
                        team_paths[node_in_team_path_index] = sub_path_new_copy

                path_index += 1
            
        self.calculate_unused_nodes(team_paths)
        return team_paths
                    
    def calculate_remove_after_cost(self, path: List[int], node_index: int) -> float:
        """
        ノードを削除した際の新しいパスコストを計算
        Args:
            path (List[int]): パス
            node_index (int): 削除するノードのインデックス
        Returns:
            float: 新しいパスコスト
        """
        path_cost = self.calculate_path_cost(path)
        change_remove_cost = self.distances[path[node_index - 1]][path[node_index + 1]] - self.distances[path[node_index - 1]][path[node_index]] - self.distances[path[node_index]][path[node_index + 1]]
        new_path_cost = path_cost + change_remove_cost
        return new_path_cost
                
    def check_node_in_team_yet(self, team_paths: List[List[int]], node: int, except_path_index: int) -> bool:
        """
        チーム内にノードが含まれているかを確認
        Args:
            team_paths (List[List[int]]): チームのパス
            node (int): ノード
            except_path_index (int): 除外するパスのインデックス
        Returns:
            bool: 含まれているか．含まれている場合はTrue，そのパスのインデックス，パス内でのノードのインデックス/含まれていない場合はFalse，-1，-1
        """
        path_index, end_path_loop = 0, len(team_paths)
        while path_index < end_path_loop:
            if path_index == except_path_index:
                path_index += 1
                continue
            else:
                path = team_paths[path_index]
                node_index, end_index = 1, len(path) - 1
                while node_index < end_index:
                    if path[node_index] == node:
                        return True, path_index, node_index
                    node_index += 1
            path_index += 1
        return False, -1, -1
        
    def calculate_add_cost(self, path: List[int], node: int, node_index: int) -> float:
        """
        ノードを追加した際のコストを計算
        Args:
            path (List[int]): パス
            node (int): 追加するノード
            node_index (int): 追加するノードの追加後のインデックス
        Returns:
            float: 追加コスト
        """
        change_add_cost = self.distances[path[node_index - 1]][node] + self.distances[node][path[node_index]] - self.distances[path[node_index - 1]][path[node_index]]
        return change_add_cost
        
    def one_point_movement_get_node_index(self, path: List[int], node: int, path_cost: float) -> bool:
        """1点移動でノードを追加した際に最適な場所（追加コストが最小）を取得

        Args:
            path (List[int]): パス（1つ）
            node (int): 挿入するノード

        Returns:
            best_index(int): 挿入する場所のインデックス
        """
        min_cost = float("inf")
        min_cost, best_index = self.calculate_insert_min_cost(path, node, None)
        path_cost_change = path_cost + min_cost
        if path_cost_change > self.t_max:
            return None
        else:
            return best_index
                
                
    def two_point_exchange(self, team_paths: List[List[int]]) -> List[List[int]]:
        """
        2-point exchangeのmain
        Args:
            team_paths (List[List[int]]): チームのパス
        Returns:
            List[List[int]]: 2-point exchange法による探索後のチームのパス
        """
        path_index, end_path_index = 0, len(team_paths)
        while path_index < end_path_index:
            path = team_paths[path_index]
            
            node_team_index, end_node_team_index = 1, len(path) - 1
            while node_team_index < end_node_team_index:
                change_flag = False
                change_flag_not_improve_score = False
                
                node_team = path[node_team_index]
                path_score = self.calculate_path_score(path)
                path_cost = self.calculate_path_cost(path)
                path_score_sub = 0  # 暫定的に良いスコアを格納．
                
                unused_node_index, end_unused_node_index = 0, len(self.unused_nodes)
                while unused_node_index < end_unused_node_index:
                    unused_node = self.unused_nodes[unused_node_index]
                    change_bool, tmp_path, best_insert_position = self.calculate_change_cost(path, path_cost, unused_node, node_team, node_team_index)
                    if change_bool:
                        tmp_path_score = self.calculate_path_score(tmp_path)
                        if tmp_path_score > path_score: # スコアが改善した場合
                            self.unused_nodes.append(node_team)
                            self.unused_nodes.remove(unused_node)
                            team_paths[path_index] = tmp_path
                            path = copy.deepcopy(tmp_path)
                            change_flag = True
                            break
                        elif tmp_path_score > path_score_sub and best_insert_position != -1:
                            path_score_sub = tmp_path_score
                            path_sub = copy.deepcopy(tmp_path)
                            add_to_unsed_node_sub = node_team
                            remove_from_unused_node_sub = unused_node
                            change_flag_not_improve_score = True
                    unused_node_index += 1
                
                if change_flag is False and change_flag_not_improve_score is True:    # 交換が行われなかったが暫定的に悪くないものを取得できた場合
                    tmp_team_score = self.calculate_team_score(team_paths)
                    tmp_team_score = tmp_team_score - path_score + path_score_sub
                    if tmp_team_score > self.record - self.deviation:
                        team_paths[path_index] = path_sub
                        path = copy.deepcopy(path_sub) # これはいらないかも
                        self.unused_nodes.append(add_to_unsed_node_sub)
                        self.unused_nodes.remove(remove_from_unused_node_sub)

                node_team_index += 1
            path_index += 1
        return team_paths
                    
    def check_change_or_not(self, team_paths_before: List[List[int]], team_paths_after: List[List[int]]) -> bool:
        """最適解がimprovement内のinnerループ内で変化したかを確認

        Args:
            team_paths_before (List[List[int]]): 改善前のパス
            team_paths_after (List[List[int]]): 改善後のパス

        Returns:
            bool: 改善していたらTrue, そうでない場合はFalse
        """
        for path_before, path_after in zip(team_paths_before, team_paths_after):
            if path_before != path_after:
                return True
        return False
    
    
    def calculate_change_cost(self, path: List[int], path_cost: float, insert_node: int, delete_node: int, delete_node_index: int) -> Tuple[bool, List[int], int]:
        """
        ノードを交換した場合のコストを計算
        Args:
            path (List[int]): パス
            path_cost (float): パスのコスト
            insert_node (int): パスに挿入するノード
            delete_node (int): パスから削除するノード
            delete_node_index (int): 削除するノードのインデックス
        Returns:
            Tuple[bool, List[int], int]: 変更があったか、変更後のパス、最適な挿入位置
        """
        path_copy = copy.deepcopy(path)
        
        # ノードを削除したときの節約コスト
        saving_cost = self.distances[path[delete_node_index - 1]][path[delete_node_index + 1]] - self.distances[path[delete_node_index - 1]][delete_node] - self.distances[delete_node][path[delete_node_index + 1]]
        path_copy.pop(delete_node_index)

        # 挿入コスト
        insert_min_cost, best_insert_position = self.calculate_insert_min_cost(path_copy, insert_node)
        
        path_cost = path_cost + saving_cost + insert_min_cost
        
        if path_cost > self.t_max:
            return False, [], -1
        else:
            path_copy.insert(best_insert_position + 1, insert_node)
            return True, path_copy, best_insert_position
        
    def remove_cost_change(self, node_index: int, path: List[int]) -> float:
        """ノードを削除した場合のコストの変化量を計算

        Args:
            node_index (int): パス内のノードのインデックス
            path (List[int]): パス（1つ）

        Returns:
            float: ノードを削除した場合のコストの変化量
        """
        saving_cost = self.distances[path[node_index - 1]][path[node_index + 1]] - self.distances[path[node_index - 1]][path[node_index]] - self.distances[path[node_index]][path[node_index + 1]]
        return saving_cost
        
    def calculate_insert_min_cost(self, path: List[int], insert_node: int) -> Tuple[float, float]:
        """ノードをパスに挿入する際の最小コストを計算

        Args:
            path (List[int]): 挿入される側のパス
            insert_node (int): 挿入されるノード

        Returns:
            min_cost(float): 挿入する場合の最小コスト
            best_index(int): 挿入する場所のインデックス
        """
        min_cost = float("inf")
        best_index = -1
        for node_index in range(len(path) - 1):
            if path[node_index] == self.start or path[node_index] == self.goal:
                continue
            cost = self.distances[path[node_index]][insert_node] + self.distances[insert_node][path[node_index + 1]] - self.distances[path[node_index]][path[node_index + 1]]
            if cost < min_cost and self.calculate_path_cost(path) + cost <= self.t_max:
                min_cost = cost
                best_index = node_index
        return min_cost, best_index
        
    def calculate_team_score(self, team_paths: List[List[int]]) -> float:
        """パスの総スコアを計算

        Args:
            team_paths (List[List[int]]): パスのリスト（チームパス）

        Returns:
            float: パスの総スコア（チームスコア）
        """
        total_score = 0
        for path in team_paths:
            total_score += self.calculate_path_score(path)
        return total_score
    
    def calculate_path_score(self, path: List[int]) -> float:
        """パスの総スコアを計算
        Args:
            path (List[int]): パス（1つ）
        Returns:
            float: パスの総スコア
        """
        score = 0
        ave = 0
        var = 0
        for node in path:
            ave += self.scores_ave[node]
            var += self.scores_var[node]
        score = self.calculate_obj_function(ave, var)
        return score
    
    def calculate_obj_function(self, ave: float, var: float) -> float:
        """目的関数を計算

        Args:
            ave (float): 平均
            var (float): 分散

        Returns:
            float: 目的関数の値
        """
        return self.alpha * ave - (1 - self.alpha) * var

    def calculate_path_ave_var(self, path: List[int]) -> Tuple[float, float]:
        """パスの平均と分散を計算

        Args:
            path (List[int]): パス（1つ）

        Returns:
            Tuple[float, float]: 平均と分散
        """
        ave = 0
        var = 0
        for node in path:
            ave += self.scores_ave[node]
            var += self.scores_var[node]
        return ave, var
    
    def create_initial_solution(self) -> List[List[int]]:
        """初期解を生成

        Returns:
            List[List[int]]: 最適解のチームパス（複数）
        """
        initial_paths = []
        initial_paths = self.create_member_paths()
        self.calculate_unused_nodes(initial_paths)
        initial_paths = self.greedy_search(initial_paths)
        return initial_paths

    def create_member_paths(self) -> List[List[int]]:
        # ノードをランダムに並び替える
        shuffled_nodes = self.nodes_in_ellipse[:]
        random.shuffle(shuffled_nodes)
        team_paths = []
        for i in range(self.member):
            if i < len(shuffled_nodes):
                # 各パスの先頭に start、1つのノード、末尾に goal を追加
                path = [self.start, shuffled_nodes[i], self.goal]
                team_paths.append(path)

        return team_paths
        
    def get_fartherst_nodes_in_ellipse(self) -> List[Tuple[int, float]]:
        """楕円内にあるノードの中で始点→ノード→終点の距離が最大となるノードの組み合わせをl(num_farthest_nodes)個取得

        Returns:
            List[Tuple[int, float]]: ノード番号とその移動コストのリスト
        """
        num_farthest_nodes = min(5, len(self.nodes_in_ellipse))
        farthest_nodes = []

        for node_num in self.nodes_in_ellipse:
            distance_from_start = self.distances[0][node_num]
            distance_from_goal = self.distances[self.goal][node_num]
            distance_node = distance_from_start + distance_from_goal

            if len(farthest_nodes) < num_farthest_nodes:
                farthest_nodes.append((node_num, distance_node))
                farthest_nodes.sort(key=lambda x: x[1], reverse=True)
            else:
                if distance_node > farthest_nodes[-1][1]:
                    farthest_nodes[-1] = (node_num, distance_node)
                    # 挿入位置を見つけて挿入
                    for i in range(num_farthest_nodes - 1, 0, -1):
                        if farthest_nodes[i][1] > farthest_nodes[i - 1][1]:
                            farthest_nodes[i], farthest_nodes[i - 1] = farthest_nodes[i - 1], farthest_nodes[i]
                        else:
                            break

        return farthest_nodes

    def greedy_search(self, initial_paths: List[int]) -> Tuple[List[List[int]], List[int]]:
        """貪欲法による探索

        Args:
            initial_paths (List[int]): 人数分の初期パス（始点→ノード→終点のみ）

        Returns:
            Tuple[List[List[int]], List[int]]: 最適解のチームパス（複数）
        """
        
        member = 0
        while member < self.member:
            
            path = initial_paths[member]
            # 既存のパスに対してノードを貪欲に挿入
            path = self.insert_nodes_greedily(path)
            initial_paths[member] = path
            self.calculate_unused_nodes(initial_paths)
            member += 1
        return initial_paths
    
    def insert_nodes_greedily(self, path: List[int]) -> bool:
        """指定されたパスに対してノードを貪欲法で挿入する

        Args:
            path (List[int]): 現在のパス．1つのパス
            ellipse_unused (List[int]): 挿入可能なノードのリスト

        Returns:
            path (List[int]): ノードを挿入した後のパス
        """
        ellipse_unused_copy = copy.deepcopy(self.unused_nodes)
        random.shuffle(ellipse_unused_copy)
        while True:
            improved = False
            best_cost = float('inf')
            best_node = None
            best_position = None

            # 各ノードを挿入する位置を見つける
            for node in ellipse_unused_copy:
                cost, i= self.calculate_insert_cost(node, path)
                if cost < best_cost:
                    best_cost = cost
                    best_node = node
                    best_position = i

            # 挿入するノードが見つからない場合は終了
            if best_node is None:
                break

            # ノードを挿入したことを仮定してパスの総コストを計算
            temp_path = path[:best_position + 1] + [best_node] + path[best_position + 1:]
            path_cost = self.calculate_path_cost(temp_path)
            if path_cost <= self.t_max:
                path.insert(best_position + 1, best_node)
                ellipse_unused_copy.remove(best_node)
                improved = True
            if improved is False:
                break
        return path
    
    def calculate_insert_cost(self, node: int, path: List[List[int]]) -> Tuple[float, int]:
        """ノードを挿入した場合のコストを計算

        Args:
            node(int): ノード番号
            paths(List[int]): パスのリスト

        Returns:
            Tuple[float, int]: 最小コストとその挿入位置
        """
        minimum_cost = float("inf")
        best_position = 0
        for i in range(len(path) - 1):
            insert_cost = 0
            insert_cost = self.distances[node][path[i]] + self.distances[node][path[i + 1]] - self.distances[path[i]][path[i + 1]]
            if insert_cost < minimum_cost:
                minimum_cost = insert_cost
                best_position = i
        return minimum_cost, best_position        
            
    def calculate_path_cost(self, path: List[int]) -> float:
        """パスの総コストを計算

        Args:
            path (List[int]): （1つの）パス

        Returns:
            float: パスの総コスト
        """
        path_cost = 0
        for node_index in range(len(path) - 1):
            path_cost += self.distances[path[node_index]][path[node_index + 1]]
        return path_cost

    def get_ellipse_nodes(self) ->  List[int]:
        """楕円内にあるノードのノード番号のみを取得．始点と終点も含む．

        Returns:
            List[int]: 楕円内にあるノード番号を格納したリスト
        """
        nodes_in_ellipse = []
        distance_from_start = self.distances[0]
        distance_from_goal = self.distances[self.goal]
        for node_num in range(self.vertex_num):
            distance_node = distance_from_start[node_num] + distance_from_goal[node_num]
            if distance_node < self.t_max:
                nodes_in_ellipse.append(node_num)
        return sorted(nodes_in_ellipse)

    def calculate_unused_nodes(self, paths):
        """
        self.nodes_in_ellipse から paths に含まれるノードを引いて self.unused_nodes を計算する。

        Args:
            paths (List[List[int]]): ノードのリストのリスト。

        Returns:
            None
        """
        # paths をフラットなリストに展開
        flattened_paths = [node for path in paths for node in path]

        # self.nodes_in_ellipse から paths のノードを引いて self.unused_nodes を更新
        self.unused_nodes = list(set(self.nodes_in_ellipse) - set(flattened_paths))

    def get_data_from_csv(self, vertex_number: str, file_id: str)    -> Tuple:
        # ディレクトリとファイル名を結合
        file_name = f"vertex_{vertex_number}_{file_id}.json"
        file_path = os.path.join("previous_dataset", file_name)
        # ファイルが存在するか確認
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"指定されたファイルが見つかりません: {file_path}")

        # JSONファイルを読み取る
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        return data
        
        
    def debug_output(self):
        """
        デバッグをしてストップした際にファイル番号を出力する関数
        """
        
        print(f"頂点数: {self.vertex_num}")
        print(f"ファイルID: {self.file_id}")
        print(f"メンバ数: {self.member}")
        print(f"t_max: {self.t_max}")
        
    def debug_start_goal(self, paths: List[List[int]])  -> None:
        """スタートとゴールが正しいかを確認

        Args:
            paths (List[List[int]]): 複数のパス
        """
        for path in paths:
            if path[0] != self.start or path[-1] != self.goal:
                print("debug: スタートまたはゴールが間違っています．")
                print(paths)
                self.debug_output()
                sys.exit()

    def debug_node_in_path(self, paths: List[List[int]])    -> None:
        """パス内のノードが全て1度しか登場していないかを確認（startとgoalは重複を許容）

        Args:
            paths (List[List[int]]): 複数のパス
        """
        all_nodes = []
        for path in paths:
            all_nodes.extend(path)

        # startとgoalを除外して重複チェック
        filtered_nodes = [node for node in all_nodes if node != self.start and node != self.goal]
        duplicates = [node for node in set(filtered_nodes) if filtered_nodes.count(node) > 1]

        if duplicates:
            print("debug: パス内のノードが重複しています（startとgoalを除外）。")
            print(f"重複しているノード: {duplicates}")
            self.debug_output()
            sys.exit()
            
    def debug_cost_t_max(self, paths: List[int]) -> None:
        """パスのコストがt_max以下かを確認

        Args:
            paths (List[int]): パス

        Returns:
            float: パスのコスト
        """
        for path in paths:
            if self.calculate_path_cost(path) > self.t_max:
                print("debug: パスのコストがt_maxを超えています．")
                self.debug_output()
                sys.exit()            
            
    def list_unique_elements(self) -> None:
        """
        リスト内の要素が一意であるかを確認
        """
        if len(self.unused_nodes) != len(set(self.unused_nodes)):
            print(self.unused_nodes)
            print("debug: リスト内の要素が一意ではありません．")
            self.debug_output()
            sys.exit()
    
    def check_all_nodes_in_ellipse(self, paths):
        """
        pathsの要素とself.unused_nodesを組み合わせて、
        self.nodes_in_ellipseの全ての要素をカバーできているかを確認する関数。

        Args:
            paths (List[List[int]]): チェック対象のパス群。

        Returns:
            bool: 全てのself.nodes_in_ellipseをカバーしていればTrue、それ以外はFalse。
        """
        # pathsの要素をフラットなリストに展開
        flattened_paths = [node for path in paths for node in path]

        # pathsの要素とself.unused_nodesを結合
        combined_nodes = set(flattened_paths + self.unused_nodes)

        # self.nodes_in_ellipseが全て含まれているか確認
        if set(self.nodes_in_ellipse).issubset(combined_nodes) is False:
            print("debug: self.nodes_in_ellipseの全ての要素がカバーされていません。")
            print("カバーされていないノードは以下の通りです。")
            print(set(self.nodes_in_ellipse) - combined_nodes)
            print(f"{self.nodes_in_ellipse=}, {self.unused_nodes=}, {paths=}")
            self.debug_output()
            sys.exit()
        
    def debug_main(self, team_paths: List[List[int]]) -> None:
        """
        デバッグ用のメイン処理
        team_paths(List[List[int]]): チームパスのリスト
        """
        print("ellipse_solver.py（提案手法2）")
        self.debug_cost_t_max(team_paths)
        self.debug_start_goal(team_paths)
        self.debug_node_in_path(team_paths)
        self.list_unique_elements()
        self.check_all_nodes_in_ellipse(team_paths)
        return team_paths

def get_numbers_from_previous_directory(directory_path: str)    -> None:
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