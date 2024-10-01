from multiprocessing import Pool


def square(x):
    return x * x


if __name__ == '__main__':
    with Pool(processes=4) as pool:  # 创建一个包含4个进程的进程池
        numbers = [1, 2, 3, 4, 5]
        results = pool.imap(square, numbers)  # 并发地计算每个元素的平方
        for result in results:
            print(result)