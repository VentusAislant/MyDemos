from abc import ABC


class A(ABC):
    def __init__(self, config):
        self.config = config
        self.is_pretraining = True
        print(f'A CLass init, config: {self.config}')


class B:
    def __init__(self, config):
        self.config = config
        super().__init__(config)
        print(f'B CLass init, config: {self.config}')

    def forward(self):
        print(f'B forward, config: {self.config}, {self.is_pretraining}')


class C(B, A):
    def __init__(self, config):
        # Attention! C has not super init
        super(B, self).__init__(config)
        print(f'C CLass init, config: {config}, {self.is_pretraining}')
        self.config = config

    def forward(self):
        super().forward()


if __name__ == '__main__':
    c = C(config={'a': 1})
    print(C.__mro__)
    c.forward()
