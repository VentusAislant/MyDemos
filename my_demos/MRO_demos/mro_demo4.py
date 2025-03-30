from abc import ABC

class A(ABC):
    def __init__(self, config):
        self.config = config
        print(f'A CLass init {self.config}')

    def forward(self):
        print('A forward')


class B:
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.config = {'name': 'Ben'}
        print(f'B CLass init, config: {self.config}')

    def show(self):
        print(f'B is show config: {self.config}.')

    def forward(self):
        print('B forward')


class C(B, A):
    def __init__(self, config=None):
        # Attention! C has not super init
        super(B, self).__init__(config)
        self.config = config
        print(f'C CLass init, config: {self.config}')

    def forward(self):
        super().forward()


if __name__ == '__main__':
    c = C(config={"name": "Rose"})
    print(C.__mro__)

    # 虽然 B 的初始化函数没有调用，但是仍然可以调用其方法
    # 由于 B 的初始化函数没有调用，self.config使用的是 C 的
    c.show()  # B is show config: {'name': 'Rose'}.

    # super().forward() 会找到 MRO 中的第一个类的 forward 方法
    c.forward()
