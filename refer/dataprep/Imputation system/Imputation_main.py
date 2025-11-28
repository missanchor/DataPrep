import warnings
# 设置警告过滤器，将所有警告都忽略
warnings.filterwarnings("ignore")
# 恢复默认的警告设置
# warnings.resetwarnings()
import pandas as pd
import re
import database
from Algorithm import GAIN, SCIS, VAEGAIN

user = 0
pwd = 0
db = 0

while True:
    ReadData_method = input("请选择读取数据的方式：\n1.从数据库中读取\n2.从csv文件中读取\n")
    if ReadData_method != '1' and ReadData_method != '2':
        print("请输入1或2来选择读取数据方式！")
        continue
    elif ReadData_method == '1':
        while True:
            user = input("请输入数据库的用户名：")
            pwd = input("请输入数据库的密码：")
            db = input("请输入要使用的数据库名称：")
            if database.connect_database(user, pwd, db):
                print("数据库连接成功！")
                break
            else:
                print("数据库连接失败！")
        while True:
            table_name = input("请输入要导入的数据表名：")
            pattern = r'^[a-zA-Z_$][a-zA-Z0-9_$]{0,63}$'
            illegal = re.match(pattern, table_name) is None
            if illegal:
                print("数据表名不合法!MySQL数据表名必须以字母、下划线（_）或美元符（$）开头，且不能超过64位！")
                continue
            df_ori = database.read_database(table_name, user, pwd, db)
            if type(df_ori) == str:
                print("未找到匹配的数据表！")
                continue
            try:
                df_ori = df_ori.astype(float)  # 转为浮点数，不然None类型会报错
            except:
                print("提示：数据类型错误！数据类型只能为数字！")
                continue
            break
        print("数据上传成功！")
        # print(df_ori)
    elif ReadData_method == '2':
        while True:
            csv_dir = input("请输入csv文件的路径：")
            try:
                df_ori = pd.read_csv(csv_dir)
            except:
                print("提示：未找到相关文件！")
                continue
            try:
                df_ori = df_ori.astype(float)  # 转为浮点数，不然None类型会报错
                break
            except:
                print("提示：数据类型错误！数据类型只能为数字！")
                continue
        print("数据上传成功！")
        # print(df_ori)
    column_name = df_ori.columns.tolist()  # 保存列名，用于后面将np数组转为df
    # df_ori = df_ori.astype(float)  # 转为浮点数，不然None类型会报错
    while True:
        Alg_select = input("请选择要使用的算法：\n1.GAIN\n2.VAEGAIN\n3.SCIS\n")
        if Alg_select != '1' and Alg_select != '2' and Alg_select != '3':
            print("请输入1或2或3来选择算法！")
            continue
        elif Alg_select == '1':
            imputed_data = GAIN.main(df_ori)
            # print(imputed_data)
            print("补全完成！")
        elif Alg_select == '2':
            imputed_data = VAEGAIN.VAE_GAIN(df_ori)
            # print(imputed_data)
            print("补全完成！")
        elif Alg_select == '3':
            imputed_data = SCIS.main(df_ori)
            # print(imputed_data)
            print("补全完成！")
        break
    df_imp = pd.DataFrame(imputed_data, columns=column_name)  # 将np数组转为df
    while True:
        op_select = input("请选择接下来的操作：\n1.打印补全后的数据\n2.将数据写入数据库\n3.将数据保存为csv文件\n4.直接退出\n")
        if op_select != '1' and op_select != '2' and op_select != '3' and op_select != '4':
            print("请输入1或2或3或4来选择接下来的操作！")
            continue
        elif op_select == '1':
            print(df_imp)
        elif op_select == '2':
            if not database.connect_database(user, pwd, db):
                while True:
                    user = input("请输入数据库的用户名：")
                    pwd = input("请输入数据库的密码：")
                    db = input("请输入要使用的数据库名称：")
                    if database.connect_database(user, pwd, db):
                        print("数据库连接成功！")
                        break
                    else:
                        print("数据库连接失败！")
            while True:
                download_tablename = input("请输入要写入的数据表名称：")
                pattern = r'^[a-zA-Z_$][a-zA-Z0-9_$]{0,63}$'
                illegal = re.match(pattern, download_tablename) is None
                if illegal:
                    print("数据表名不合法!MySQL数据表名必须以字母、下划线（_）或美元符（$）开头，且不能超过64位！")
                    continue
                exist_flag = 0
                if database.is_table_exists(download_tablename, user, pwd, db):
                    print("该数据表已存在！请选择接下来的操作：")
                    while True:
                        op2_select = input("1.将其覆盖\n2.创建副本\n3.重新输入数据表名\n")
                        if op2_select != '1' and op2_select != '2' and op2_select != '3':
                            print("请输入1或2或3来选择接下来的操作！")
                            continue
                        elif op2_select == '1':
                            exist_flag = 1
                            break
                        elif op2_select == '2':
                            exist_flag = 2
                            break
                        elif op2_select == '3':
                            exist_flag = 3
                            break
                if exist_flag == 0 or exist_flag == 1:
                    database.write_to_database(df_imp, download_tablename, 'cover', user, pwd, db)
                    print("数据成功写入数据库！")
                    break
                elif exist_flag == 2:
                    database.write_to_database(df_imp, download_tablename, 'copy', user, pwd, db)
                    print("数据成功写入数据库！")
                    break
                else:
                    continue
        elif op_select == '3':
            csv_name = input("请输入要保存为的csv文件名：")
            df_imp.to_csv(csv_name, index=False)
            print("数据成功保存为csv文件！")
        elif op_select == '4':
            break
    break

