import os

if __name__ == '__main__':
    work_path = os.getcwd()

    # checkpoints 下的目录层级
    if not os.path.exists(work_path + "\\dialogue\\checkpoints\\tensorflow"):
        os.mkdir(work_path + "\\dialogue\\checkpoints\\tensorflow")
    if not os.path.exists(work_path + "\\dialogue\\checkpoints\\pytorch"):
        os.mkdir(work_path + "\\dialogue\\checkpoints\\pytorch")

    # data下的目录层级
    if not os.path.exists(work_path + "\\dialogue\\data\\history"):
        os.mkdir(work_path + "\\dialogue\\data\\history")
    if not os.path.exists(work_path + "\\dialogue\\data\\preprocess"):
        os.mkdir(work_path + "\\dialogue\\data\\preprocess")
    if not os.path.exists(work_path + "\\dialogue\\data\\pytorch"):
        os.mkdir(work_path + "\\dialogue\\data\\pytorch")

    # model 下的目录层级
    if not os.path.exists(work_path + "\\dialogue\\models\\tensorflow"):
        os.mkdir(work_path + "\\dialogue\\models\\tensorflow")
    if not os.path.exists(work_path + "\\dialogue\\models\\pytorch"):
        os.mkdir(work_path + "\\dialogue\\models\\pytorch")
