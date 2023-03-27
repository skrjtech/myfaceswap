import time 
from concurrent.futures import ThreadPoolExecutor

def Worker(*args):
    print("Start")
    time.sleep(5)
    print("End")

def main():
    try:
        with ThreadPoolExecutor(max_workers=8) as exec:
            exec.map(Worker, [None])
            exec.map(Worker, [None])

    except:
        print("finish")


if __name__ == '__main__':
    main()