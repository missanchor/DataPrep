'''
                            算法文件示例:
在算法文件中，
(1) 你需要包含一个main()函数，它将被系统调用。
(2) 你可以导入任何类型的模块或包。
(3) 你可以定义任何类型的辅助函数。
(4) 你的main()函数应该只包含两个参数:
    第一个是输入的数据，类型是pandas.DataFrame。
    第二个参数的类型是parameter()，也就是在参数文件中定义的类型。
    在参数类中，你可以封装尽可能多的你需要的参数。
(5) main()函数的输出类型应该是pandas.DataFrame。
'''

import pandas as pd

def main(data, args):
    for column in list(data.columns[data.isnull().sum() > 0]):
        try:
            mean_value = data[column].mean()
            data[column].fillna(mean_value*args.weight, inplace=True)
        except:
            data[column].fillna(value=args.fill_value, axis=0, inplace=True)
    return data
    
