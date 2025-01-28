# ruff: noqa: F401
import json
import os

import japanize_matplotlib  # 日本語化ライブラリ
import matplotlib.pyplot as plt
import pandas as pd


def plot_from_json_compare_cef_method(member: int, t_max_ratio: float, probability: float) -> None:
    # JSONファイルのパスを定義
    json_file_path = 'graph_plot/compare/compare_cef_method.json'

    # JSONファイルからデータを読み込む
    with open(json_file_path, "r") as file:
        data = json.load(file)

    # 保存先ディレクトリが存在しない場合は作成
    save_dir = "graph_plot/compare"
    os.makedirs(save_dir, exist_ok=True)

    # vertex_numとfile_idのリストを生成
    vertex_num_list = [f"{i:03}" for i in range(15, 26)]
    file_id_list = [f"{i:03}" for i in range(0, 101)]

    # データをDataFrameに変換
    df = pd.DataFrame(data)

    # 手法ごとの線スタイルを定義
    line_styles = {
        "CEF BY DP": '-',
        "Proposed CEF(\u03b1=0.01)": '-.',
        "Proposed CEF Step (\u03b1=0.1)": ':'
    }

    # nameのマッピング
    name_mapping = {
        "CEF BY DP": "CEF BY DP",
        "Proposed Ellipse CEF alpha hundred": "Proposed CEF(\u03b1=0.01)",
        "Proposed Ellipse CEF alpha one": "Proposed CEF Step (\u03b1=0.1)"
    }

    for vertex_num in vertex_num_list:
        for file_id in file_id_list:
            # 指定条件と一致するデータを抽出
            filtered_df = df[
                (df['member'] == member) &
                (df['t_max_ratio'] == t_max_ratio) &
                (df['probability'] == probability) &
                (df['vertex_num'] == vertex_num) &
                (df['file_id'] == file_id)
            ]

            if filtered_df.empty:
                continue

            # name列をマッピング
            filtered_df['name'] = filtered_df['name'].replace(name_mapping)

            # グラフ作成
            plt.figure(figsize=(8, 6))
            for name, group in filtered_df.groupby('name'):
                cef_data = group['cef'].values[0]  # cef列のデータを取得
                m_values, v_values = zip(*cef_data)

                # m_valuesでソートし、対応するv_valuesも並べ替え
                sorted_pairs = sorted(zip(m_values, v_values))  # ペアをソート
                m_values, v_values = zip(*sorted_pairs)  # 再分解
                if name == "CEF BY DP":
                    plt.plot(m_values, v_values, marker='o', label=name, linestyle=line_styles.get(name, '-'), color = 'black')
                else:
                    plt.plot(m_values, v_values, marker='o', label=name, linestyle=line_styles.get(name, '-'), alpha=0.6)

            # グラフの設定
            plt.xlabel("通過したノードの報酬の平均の総和$m_y$", fontsize=14)
            plt.ylabel("通過したノードの報酬の分散の総和$v_y$", fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True)

            # グラフをファイルに保存
            file_name = f"cef_{member}_{vertex_num}_{file_id}.png"
            plt.savefig(os.path.join(save_dir, file_name))
            plt.close()

    # プロセス完了メッセージを表示
    print(f"Plots have been created and saved in {save_dir} as PNG files.")





if __name__ == '__main__':
    plot_from_json_compare_cef_method(2, 0.3, 0.8)