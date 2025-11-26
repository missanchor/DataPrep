import pandas as pd
import pymysql
import re

host = "127.0.0.1"
port = 3306
# user='root'
# pwd='root'
charset='utf8'
# database='semi'


def read_database(table_name, user, pwd, database):
    # 创建连接
    conn = pymysql.connect(host=host, port=port, user=user, passwd=pwd, charset=charset, db=database)
    # 创建游标
    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
    try:
        query = 'select * from {}'.format(table_name)
        df = pd.read_sql(query, conn)
        # df.drop(df.columns[0], axis=1, inplace=True)  # 删除id0列
        # print(df)
        cursor.close()
        conn.close()
        return df
    except Exception as eee:
        # print(eee)
        cursor.close()
        conn.close()
        return 'No Match Table'
# print(read_database('new_weather'))


def is_table_exists(table_name, user, pwd, database):
    # 创建连接
    conn = pymysql.connect(host=host, port=port, user=user, passwd=pwd, charset=charset, db=database)
    # 创建游标
    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
    cursor.execute("SHOW TABLES;")
    tables = cursor.fetchall()
    table_names = [table['Tables_in_'+database] for table in tables]
    for i in range(len(table_names)):
        if table_names[i] == table_name:
            cursor.close()
            conn.close()
            return True
    cursor.close()
    conn.close()
    return False
# print(is_table_exists('new_weather'))


def write_to_database(df_from_system, table_name, cover_or_copy, user, pwd, database):
    # 创建连接
    conn = pymysql.connect(host=host, port=port, user=user, passwd=pwd, charset=charset, db=database)
    # 创建游标
    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
    # 获取数据框的标题行（即字段名称）,将来作为sql语句中的字段名称。
    df = df_from_system
    df = df.where(df.notnull(), None)
    columns = df.columns.tolist()
    # print(columns)
    # print(f.values.tolist())

    # 将csv文件中的字段类型转换成mysql中的字段类型
    # types = f.dtypes
    field = []  # 用来接收字段名称的列表
    field_type = []  # 用来接收字段名称和字段类型的列表
    for item in columns:
        # transform the column_name to the string that only contains the alphelt, number, and _
        new_item = re.sub(r'[^A-Za-z0-9]', '_', item)
        # print(new_item)
        if 'int' in str(df[item].dtype):
            char = new_item + ' INT'
        elif 'float' in str(df[item].dtype):
            char = new_item + ' FLOAT'
        elif 'object' in str(df[item].dtype):
            char = new_item + ' VARCHAR(255)'
        elif 'datetime' in str(df[item].dtype):
            char = new_item + ' DATETIME'
        else:
            char = new_item + ' VARCHAR(255)'
        field_type.append(char)
        field.append(new_item)
    # 将table列表中的元素用逗号连接起来，组成table_sql语句中的字段名称和字段类型片段，用来创建表。
    fields = ','.join(field)
    field_types = ','.join(field_type)
    # print(fields)
    # print(field_types)

    if cover_or_copy == 'cover':
        # 如果数据库表已经存在，首先删除它
        cursor.execute('drop table if exists {};'.format(table_name))
        conn.commit()
        # 构建创建数据表的SQL语句
        # create_table_sql = 'CREATE TABLE IF NOT EXISTS ' + table_name + '(' + 'id0 int PRIMARY KEY NOT NULL auto_increment,' + field_types + ');'
        create_table_sql = 'CREATE TABLE IF NOT EXISTS ' + table_name + '(' + field_types + ');'
        # 开始创建数据表
        cursor.execute(create_table_sql)
        conn.commit()
        # 将数据框的数据读入列表。每行数据是一个列表，所有数据组成一个大列表。也就是列表中的列表，将来可以批量插入数据库表中。
        values = df.values.tolist()  # 所有的数据
        # print(values)
        # 计算数据框中总共有多少个字段，每个字段用一个 %s 替代。
        s = ','.join(['%s' for _ in range(len(df.columns))])
        # 构建插入数据的SQL语句
        insert_sql = 'insert into {}({}) values({})'.format(table_name, fields, s)
        # 使用 executemany批量插入数据
        cursor.executemany(insert_sql, values)
        conn.commit()
    elif cover_or_copy == 'copy':
        # 如果数据库表的副本已经存在，首先删除它
        cursor.execute('drop table if exists {};'.format(table_name + '_副本'))
        conn.commit()
        # 构建创建数据表的SQL语句
        create_table_sql = 'CREATE TABLE IF NOT EXISTS ' + table_name + '_副本' + '(' + field_types + ');'
        # 开始创建数据表
        cursor.execute(create_table_sql)
        conn.commit()
        # 将数据框的数据读入列表。每行数据是一个列表，所有数据组成一个大列表。也就是列表中的列表，将来可以批量插入数据库表中。
        values = df.values.tolist()  # 所有的数据
        # print(values)
        # 计算数据框中总共有多少个字段，每个字段用一个 %s 替代。
        s = ','.join(['%s' for _ in range(len(df.columns))])
        # 构建插入数据的SQL语句
        insert_sql = 'insert into {}({}) values({})'.format(table_name + '_副本', fields, s)
        # 使用 executemany批量插入数据
        cursor.executemany(insert_sql, values)
        conn.commit()


    cursor.close()
    conn.close()
    # 返回写入数据个数
    # return int(f.shape[0])
    return "write to database successfully"


def connect_database(user, pwd, database):  # 判断数据库是否能成功连接
    # 创建连接
    try:
        conn = pymysql.connect(host=host, port=port, user=user, passwd=pwd, charset=charset, db=database)
        # 创建游标
        cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
        cursor.close()
        conn.close()
        return True
    except:
        return False
# print(connect_database('root', 'root', 'semi'))


def csv_to_database(file_path, table_name, user, pwd, database):
    # 创建连接
    conn = pymysql.connect(host=host, port=port, user=user, passwd=pwd, charset=charset, db=database)
    # 创建游标
    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
    df = pd.read_csv(file_path, encoding="utf-8")
    df = df.where(df.notnull(), None)
    # rows_null = df.isnull().sum(axis=1)  # 统计每行空值的数量，保存在rows_null列中
    # df['empty_num'] = rows_null  # 将rows_null作为新的一列加入到df中，列名为empty_num
    columns = df.columns.tolist()
    # print(columns)
    # print(f.values.tolist())

    # 将csv文件中的字段类型转换成mysql中的字段类型
    # types = f.dtypes
    field = []  # 用来接收字段名称的列表
    field_type = []  # 用来接收字段名称和字段类型的列表
    for item in columns:
        # transform the column_name to the string that only contains the alphelt, number, and _
        new_item = re.sub(r'[^A-Za-z0-9]', '_', item)
        # print(new_item)
        if 'int' in str(df[item].dtype):
            char = new_item + ' INT'
        elif 'float' in str(df[item].dtype):
            char = new_item + ' FLOAT'
        elif 'object' in str(df[item].dtype):
            char = new_item + ' VARCHAR(255)'
        elif 'datetime' in str(df[item].dtype):
            char = new_item + ' DATETIME'
        else:
            char = new_item + ' VARCHAR(255)'
        field_type.append(char)
        field.append(new_item)
    # 将table列表中的元素用逗号连接起来，组成table_sql语句中的字段名称和字段类型片段，用来创建表。
    fields = ','.join(field)
    field_types = ','.join(field_type)
    # print(fields)
    # print(field_types)

    # 如果数据库表已经存在，首先删除它
    cursor.execute('drop table if exists {};'.format(table_name))
    conn.commit()
    # 构建创建数据表的SQL语句
    #create_table_sql = 'CREATE TABLE IF NOT EXISTS ' + table_name + '(' + 'id0 int PRIMARY KEY NOT NULL auto_increment,' + field_types + ');'
    create_table_sql = 'CREATE TABLE IF NOT EXISTS ' + table_name + '(' + field_types + ');'
    # 开始创建数据表
    cursor.execute(create_table_sql)
    conn.commit()

    # 将数据框的数据读入列表。每行数据是一个列表，所有数据组成一个大列表。也就是列表中的列表，将来可以批量插入数据库表中。
    values = df.values.tolist()  # 所有的数据
    # print(values)
    # 计算数据框中总共有多少个字段，每个字段用一个 %s 替代。
    s = ','.join(['%s' for _ in range(len(df.columns))])
    # 构建插入数据的SQL语句
    insert_sql = 'insert into {}({}) values({})'.format(table_name, fields, s)
    # 使用 executemany批量插入数据
    cursor.executemany(insert_sql, values)
    conn.commit()

    cursor.close()
    conn.close()
    # 返回写入数据个数
    # return int(f.shape[0])
    return "csv to database successfully"
# print(csv_to_database('C:\\Users\\wo\\Desktop\\Imputation system\\csv_test\\new_weather100.csv','new_weather100_1', 'root', 'root', 'semi'))
