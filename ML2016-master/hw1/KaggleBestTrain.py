from sys import argv
import math
import numpy as np
from collections import deque

def loadTrainingData(Filename):
    TrainDataFile = open(Filename, "r")
    lines = TrainDataFile.readlines()
    TrainDataFile.close()

    data = []
    for m in range(1, len(lines), 18 * 20):
        X = [[] for _ in range(18)]
        for d in range(0, 18 * 20, 18):
            for i in range(18):
                row = lines[m + d + i].rstrip('\r\n').split(',')
                X[i].extend(row[3:])

        for i in range(len(X[10])):
            if X[10][i] == "NR":
                X[10][i] = 0.0

        for row in X:
            for i in range(len(row)):
                row[i] = float(row[i])

        d = deque()
        for i in range(9):
            d.append([X[j][i] for j in range(18)])
        for i in range(9, 20 * 24):
            data.append((np.array(d), X[9][i]))
            d.popleft()
            d.append([X[j][i] for j in range(18)])

    return data

def Normalize(X):
    S = np.zeros(X[0][0].shape)
    Y = 0
    for x, y in X:
        S += x
        Y += y
    S /= len(X)
    Y /= len(X)
    dS = np.zeros(S.shape)
    dY = 0
    for x, y in X:
        dS += (x - S) ** 2
        dY += (y - Y) ** 2
    dS = np.sqrt(dS / len(X))
    dY = math.sqrt(dY / len(X))
    for i in range(len(X)):
        X[i] = ((X[i][0] - S) / dS, (X[i][1] - Y) / dY)
    return S, dS, Y, dY

def loadCoefficient(Filename):
    File = open(Filename, "r")
    lines = File.readlines()
    File.close()
    B = float(lines[0].rstrip("\r\n"))
    Input = []
    for i in range(1, 10):
        Input.append([float(s) for s in lines[i].rstrip("\r\n").split(",")])
    C = np.array(Input)
    return B, C

def saveCoefficient(Filename, B, C, normalized_attr):
    File = open(Filename, "w")
    File.write(str(B))
    File.write("\n")
    for row in C:
        File.write(",".join([str(e) for e in row]))
        File.write("\n")
    for row in normalized_attr[0]:
        File.write(",".join([str(e) for e in row]))
        File.write("\n")
    for row in normalized_attr[1]:
        File.write(",".join([str(e) for e in row]))
        File.write("\n")
    File.write('%f,%f\n' % (normalized_attr[2], normalized_attr[3]))
    File.close()

def F(B, C, X):
    return (C * X).sum() + B

def Loss(B, C, X):
    W = 0.0
    for x, y in X:
        Y = F(B, C, x)
        W += (y - Y) ** 2
    return W / len(X)

def Gradient(B, C, x, y):
    Y = F(B, C, x)
    return (Y - y), (Y - y) * x

def main():
    X = loadTrainingData(argv[1])
    normalized_attr = Normalize(X)
    np.random.seed(3902082)
    np.random.shuffle(X)
    B, C = loadCoefficient(argv[2])
    if len(argv) > 3:
        Iterations = int(argv[3])
    else:
        Iterations = 10000

    Alpha = 0.01
    AccuGradB = 1e-20
    AccuGradC = np.full((9, 18), 1e-20)
    for _ in range(Iterations):
        if _ % 100 == 0:
            print (math.sqrt(Loss(B, C, X)) * normalized_attr[3])
            saveCoefficient("coefficient_best.csv", B, C, normalized_attr)

        GradB = 0.0
        GradC = np.zeros((9, 18))
        for x, y in X:
            g = Gradient(B, C, x, y)
            GradB += g[0]
            GradC += g[1]
        GradB /= len(X)
        GradC /= len(X)
        AccuGradB += GradB ** 2
        AccuGradC += GradC ** 2
        B -= Alpha * GradB / math.sqrt(AccuGradB)
        C -= Alpha * GradC / np.sqrt(AccuGradC)

    saveCoefficient("coefficient_best.csv", B, C, normalized_attr)

if __name__ == "__main__":
    main()
