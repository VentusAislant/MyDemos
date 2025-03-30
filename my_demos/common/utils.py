TOTAL_PRINT_LEN = 90


def print_title(title):
    pl_total = TOTAL_PRINT_LEN - 10 - len(title) - 4
    pl1 = pl_total // 2
    pl2 = pl_total - pl1
    print(f'|+_+|{"=" * pl1}| {title} |{"=" * pl2}|+_+|')


def print_title_end():
    print(f'|+_+|{"=" * (TOTAL_PRINT_LEN - 10)}|+_+|')


if __name__ == '__main__':
    print_title('ChatBot12345')
    print_title_end()
