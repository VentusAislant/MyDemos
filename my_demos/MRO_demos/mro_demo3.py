class A:
    def __init__(self, config):
        super().__init__()
        print(f'A CLass init {config}')


class B:
    def __init__(self, config):
        super().__init__(config)
        print(f'B CLass init, config: {config}')


class C(B, A):
    def __init__(self, config):
        # Attention! C has not super init
        super().__init__(config)
        print(f'C CLass init, config: {config}')


if __name__ == '__main__':
    c = C(config={"name": "Rose"})
    print(C.__mro__)

    # 可以传递参数，但是必须保证 MRO 待调用的类的 init 函数可以接受相应的参数