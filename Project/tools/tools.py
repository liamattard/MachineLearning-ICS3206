import os

def writeNegativeTxt():
    with open('neg.txt', 'w') as f:
        for filename in os.listdir('./Dataset/non_faces'):
            f.write('Dataset/non_faces' + filename + '\n')

writeNegativeTxt()