if __name__ == '__main__':
    class Obj():
        count: int = 0
        def __init__(self):
            Obj.count += 1
            self.count = Obj.count
    O1 = Obj()
    O2 = Obj()
    O3 = Obj()
    print(O1.count)
    print(O2.count)
    print(O3.count)
    print(O1.count)
    print(O3.__class__.__name__)