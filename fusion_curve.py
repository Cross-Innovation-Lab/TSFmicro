import re

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

# samm cas2 use
def analyze_log(file_path):
    # 初始化变量
    acc_list = []  # 存储每个num_sub的ACC值
    f1_list = []   # 存储每个num_sub的F1值
    uar_list = []  # 存储每个num_sub的UAR值
    current_num_sub = None
    all_corr = 0
    all_val = 0
    f1_all = 0.0
    uar_all = 0.0

    # 正则表达式模式
    num_sub_pattern = re.compile(r'\[..........(\d+)\]')
    all_corr_pattern = re.compile(r'\[ALL_corr\]: (\d+)')
    all_val_pattern = re.compile(r'\[ALL_val\]: (\d+)')
    f1_all_pattern = re.compile(r'\[F1_ALL\]: ([\d.]+)')
    uar_all_pattern = re.compile(r'\[UAR_ALL\]: ([\d.]+)')

    # 读取日志文件
    with open(file_path, 'r') as file:
        for line in file:
            # 匹配当前num_sub
            num_sub_match = num_sub_pattern.search(line)
            if num_sub_match:
                if current_num_sub is not None:
                    # 如果当前num_sub已经存在，保存上一个num_sub的结果
                    acc = all_corr / all_val if all_val > 0 else 0.0
                    acc_list.append(acc)
                    f1_list.append(f1_all)
                    uar_list.append(uar_all)
                current_num_sub = int(num_sub_match.group(1))
                all_corr = 0
                all_val = 0
                f1_all = 0.0
                uar_all = 0.0

            # 匹配ALL_corr和ALL_val
            all_corr_match = all_corr_pattern.search(line)
            all_val_match = all_val_pattern.search(line)
            f1_all_match = f1_all_pattern.search(line)
            uar_all_match = uar_all_pattern.search(line)

            if all_corr_match:
                all_corr = int(all_corr_match.group(1))
            if all_val_match:
                all_val = int(all_val_match.group(1))
            if f1_all_match:
                f1_all = float(f1_all_match.group(1))
            if uar_all_match:
                uar_all = float(uar_all_match.group(1))
                print(f"Matched UAR: {uar_all}")  # 调试信息

        # 保存最后一个num_sub的结果
        if current_num_sub is not None:
            acc = all_corr / all_val if all_val > 0 else 0.0
            acc_list.append(acc)
            f1_list.append(f1_all)
            uar_list.append(uar_all)

    # 将ACC、F1和UAR值转换为百分比形式，并保留两位小数
    acc_list = [f"{acc * 100:.2f}" for acc in acc_list]
    f1_list = [f"{f1 * 100:.2f}" for f1 in f1_list]
    uar_list = [f"{uar * 100:.2f}" for uar in uar_list]

    return acc_list, f1_list, uar_list

#cas3 use
# def analyze_log(file_path):
#     # 初始化变量
#     f1_list = []   # 存储每个num_sub的F1值
#     uar_list = []  # 存储每个num_sub的UAR值
#     current_num_sub = None
#     f1_all = 0.0
#     uar_all = 0.0
#
#     # 正则表达式模式
#     num_sub_pattern = re.compile(r'\[..........spNO.(\d+)\]')
#     f1_all_pattern = re.compile(r'\[F1_ALL\]: ([\d.]+)')
#     uar_all_pattern = re.compile(r'\[UAR_ALL\]:([\d.]+)')
#
#     # 读取日志文件
#     with open(file_path, 'r') as file:
#         for line in file:
#             # 匹配当前num_sub
#             num_sub_match = num_sub_pattern.search(line)
#             if num_sub_match:
#                 if current_num_sub is not None:
#                     # 如果当前num_sub已经存在，保存上一个num_sub的结果
#                     f1_list.append(f1_all)
#                     uar_list.append(uar_all)
#                 current_num_sub = int(num_sub_match.group(1))
#                 f1_all = 0.0
#                 uar_all = 0.0
#
#             # 匹配F1_ALL和UAR_ALL
#             f1_all_match = f1_all_pattern.search(line)
#             uar_all_match = uar_all_pattern.search(line)
#
#             if f1_all_match:
#                 f1_all = float(f1_all_match.group(1))
#             if uar_all_match:
#                 uar_all = float(uar_all_match.group(1))
#
#         # 保存最后一个num_sub的结果
#         if current_num_sub is not None:
#             f1_list.append(f1_all)
#             uar_list.append(uar_all)
#
#     # 将F1和UAR值转换为百分比形式，并保留两位小数
#     f1_list = [f"{f1 * 100:.2f}" for f1 in f1_list]
#     uar_list = [f"{uar * 100:.2f}" for uar in uar_list]
#
#     return f1_list, uar_list
# samm cas2 use
def plot_curves(acc_list, f1_list, num_sub_list):
    # 确保数据类型为数值类型
    num_sub_list = np.arange(len(num_sub_list))
    acc_list = np.array(acc_list, dtype=float)
    f1_list = np.array(f1_list, dtype=float)

    # 拟合曲线
    acc_interp = interp1d(num_sub_list, acc_list, kind='cubic', fill_value="extrapolate")
    f1_interp = interp1d(num_sub_list, f1_list, kind='cubic', fill_value="extrapolate")

    # 生成拟合后的x值
    x = np.linspace(num_sub_list.min(), num_sub_list.max(), 300)  # 使用更平滑的x值

    # 绘制曲线图
    plt.figure(figsize=(10, 6))

    # 绘制ACC曲线
    plt.plot(x, acc_interp(x), label='ACC', color='#F8CECC')
    plt.scatter(num_sub_list, acc_list, color='#F27773', label='ACC Data Points')

    # 绘制F1曲线
    plt.plot(x, f1_interp(x), label='UF1', color='#A9C4EB')
    plt.scatter(num_sub_list, f1_list, color='#7EA6E0', label='UF1 Data Points')

    # 添加标题和标签
    plt.title('ACC and UF1 Curves')
    plt.xlabel('Num_sub')
    plt.ylabel('Percentage (%)')
    plt.legend()

    # 显示图表
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# def plot_curves(acc_list, f1_list, num_sub_list):
#     # 确保数据类型为数值类型
#     num_sub_list = np.arange(len(num_sub_list))
#     acc_list = np.array(acc_list, dtype=float)
#     f1_list = np.array(f1_list, dtype=float)
#
#     # 拟合曲线
#     acc_interp = interp1d(num_sub_list, acc_list, kind='cubic', fill_value="extrapolate")
#     f1_interp = interp1d(num_sub_list, f1_list, kind='cubic', fill_value="extrapolate")
#
#     # 生成拟合后的x值
#     x = np.linspace(num_sub_list.min(), num_sub_list.max(), 300)  # 使用更平滑的x值
#
#     # 绘制曲线图
#     plt.figure(figsize=(10, 6))
#
#     # 绘制ACC曲线
#     plt.plot(x, acc_interp(x), label='UAR', color='#F8CECC')
#     plt.scatter(num_sub_list, acc_list, color='#F27773', label='UAR Data Points')
#
#     # 绘制F1曲线
#     plt.plot(x, f1_interp(x), label='UF1', color='#A9C4EB')
#     plt.scatter(num_sub_list, f1_list, color='#7EA6E0', label='UF1 Data Points')
#
#     # 添加标题和标签
#     plt.title('UAR and UF1 Curves')
#     plt.xlabel('Num_sub')
#     plt.ylabel('Percentage (%)')
#     plt.legend()
#
#     # 显示图表
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


def main():
    # num_sub_list = ['6','7','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','25','26','28','30','31','32','33','34','35','37'] #SAMM_C5
    # num_sub_list = ['6','7','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','26','28','30','31','32','33','34','35','36','37'] #SAMM_C3
    num_sub_list = ['17', '26', '16', '9', '5', '24', '2', '13', '4', '23', '11', '12', '8', '14', '3', '19', '1', '18', '10', '20', '21', '22', '15', '6', '25', '7'] #cas2_c5
    # num_sub_list = ['17', '26', '16', '9', '5', '24', '2', '13', '4', '23', '11', '12', '8', '14', '3', '19', '1', '10', '20', '21', '22', '15', '6', '25', '7'] #cas2_c3
    # num_sub_list = ['spNO.1', 'spNO.10', 'spNO.11', 'spNO.12', 'spNO.13', 'spNO.138', 'spNO.139', 'spNO.14', 'spNO.142', 'spNO.143',
    #         'spNO.144', 'spNO.145', 'spNO.146', 'spNO.147', 'spNO.148', 'spNO.149', 'spNO.15', 'spNO.150', 'spNO.152',
    #         'spNO.153', 'spNO.154', 'spNO.155', 'spNO.156', 'spNO.157', 'spNO.158', 'spNO.159', 'spNO.160', 'spNO.161',
    #         'spNO.162', 'spNO.163', 'spNO.165', 'spNO.166', 'spNO.167', 'spNO.168', 'spNO.169', 'spNO.17', 'spNO.170',
    #         'spNO.171', 'spNO.172', 'spNO.173', 'spNO.174', 'spNO.175', 'spNO.176', 'spNO.177', 'spNO.178', 'spNO.179',
    #         'spNO.180', 'spNO.181', 'spNO.182', 'spNO.183', 'spNO.184', 'spNO.185', 'spNO.186', 'spNO.187', 'spNO.188',
    #         'spNO.189', 'spNO.190', 'spNO.192', 'spNO.193', 'spNO.194', 'spNO.195', 'spNO.196', 'spNO.197', 'spNO.198',
    #         'spNO.2', 'spNO.200', 'spNO.201', 'spNO.202', 'spNO.203', 'spNO.204', 'spNO.206', 'spNO.207', 'spNO.208',
    #         'spNO.209', 'spNO.210', 'spNO.211', 'spNO.212', 'spNO.213', 'spNO.214', 'spNO.215', 'spNO.216', 'spNO.217',
    #         'spNO.3', 'spNO.39', 'spNO.4', 'spNO.40', 'spNO.41', 'spNO.42', 'spNO.5', 'spNO.6', 'spNO.7', 'spNO.77',
    #         'spNO.8', 'spNO.9'] #cas3_c7
    # num_sub_list = ['spNO.1', 'spNO.10', 'spNO.11', 'spNO.12', 'spNO.13', 'spNO.138', 'spNO.139', 'spNO.14', 'spNO.142', 'spNO.143',
    #         'spNO.144', 'spNO.145', 'spNO.146', 'spNO.147', 'spNO.148', 'spNO.149', 'spNO.15', 'spNO.150', 'spNO.152',
    #         'spNO.153', 'spNO.154', 'spNO.155', 'spNO.156', 'spNO.157', 'spNO.158', 'spNO.159', 'spNO.160', 'spNO.161',
    #         'spNO.162', 'spNO.163', 'spNO.165', 'spNO.166', 'spNO.167', 'spNO.168', 'spNO.169', 'spNO.17', 'spNO.170',
    #         'spNO.171', 'spNO.172', 'spNO.173', 'spNO.174', 'spNO.175', 'spNO.176', 'spNO.177', 'spNO.178', 'spNO.179',
    #         'spNO.180', 'spNO.181', 'spNO.182', 'spNO.183', 'spNO.184', 'spNO.185', 'spNO.186', 'spNO.187', 'spNO.188',
    #         'spNO.189', 'spNO.190', 'spNO.192', 'spNO.193', 'spNO.194', 'spNO.195', 'spNO.196', 'spNO.197', 'spNO.198',
    #         'spNO.2', 'spNO.200', 'spNO.201', 'spNO.202', 'spNO.203', 'spNO.204', 'spNO.206', 'spNO.207', 'spNO.208',
    #         'spNO.209', 'spNO.210', 'spNO.211', 'spNO.212', 'spNO.213', 'spNO.214', 'spNO.215', 'spNO.216', 'spNO.217',
    #         'spNO.3', 'spNO.39', 'spNO.4', 'spNO.40', 'spNO.41', 'spNO.42', 'spNO.5', 'spNO.6', 'spNO.7', 'spNO.77',
    #         'spNO.8', 'spNO.9'] #cas3_c4
    # log_file_path = r'C:\Users\12171\Desktop\retmer\supplement\coding\classes\samm\samm_c5.log'  #samm_c5
    # log_file_path = r'C:\Users\12171\Desktop\retmer\supplement\coding\fusion\samm\S_to_T.log'  #samm_c5
    log_file_path = r'C:\Users\12171\Desktop\retmer\supplement\coding\fusion\cas2\S_to_T_c5.log'  # cas2_c5
    # log_file_path = r'C:\Users\12171\Desktop\retmer\supplement\coding\classes\cas2\cas2_c3.log'  # cas2_c3
    # log_file_path = r'C:\Users\12171\Desktop\retmer\supplement\coding\classes\cas3\cas3_c4.log'  # cas3_c4
    # log_file_path = r'C:\Users\12171\Desktop\retmer\supplement\coding\classes\cas3\cas3_c7.log'  # cas3_c7
    acc_list, f1_list, uar_list = analyze_log(log_file_path)
    plot_curves(acc_list, f1_list, num_sub_list)

    # 打印结果
    print("ACC列表:", acc_list)
    print("F1列表:", f1_list)
    print("UAR列表:", uar_list)

if __name__ == "__main__":
    main()