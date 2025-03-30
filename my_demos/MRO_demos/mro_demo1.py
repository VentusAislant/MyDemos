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
        super().__init__()
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
    # (<class '__main__.E'>, <class '__main__.D'>, <class '__main__.C'>,
    # <class '__main__.B'>, <class '__main__.A'>, <class 'object'>)
    print(E.__mro__)

    # 可以看出，如果每个父类都有 super().__init__()的话， MRO 是所有父类从右到左的调用方式
