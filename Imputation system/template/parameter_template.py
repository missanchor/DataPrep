'''
                            参数文件示例:
在参数文件中,
(1) 你需要定义一个名为parameter()的类，它将被算法中定义的main()函数所调用。
(2) 你需要在parameter()类的__init__()函数中定义你所需的参数的值。
'''

class parameter():
    def __init__(self):
        self.weight = 1             # weight for the column mean value
        self.fill_value = 'NONE'    # value to be filled for non-numerical value