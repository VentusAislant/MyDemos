class A:
    def __init__(self):
        super().__init__()
        print('A CLass init')


class B:
    def __init__(self):
        super().__init__()
        print('B CLass init')


class C:
    def __init__(self):
        # Attention! C has not super init
        # 如果不写 super().__init__(), MRO 初始化时会从这里断开
        print('C CLass init')


class D:
    def __init__(self):
        super().__init__()
        print('D CLass init')


class E(D, C, B, A):
    def __init__(self):
        super().__init__()
        print('E CLass init')


if __name__ == '__main__':
    e = E()
    print(E.__mro__)

    # 只有 C，D，E初始化了，到C的时候 MRO 初始化断开，不会去继续执行B的初始化
