import os

for Dirs, Subs, Files in os.walk("DirA"):
    print(Dirs, Subs, Files)
    # for file in Files:
    #     print(os.path.join(Dirs, file))