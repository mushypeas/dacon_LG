train = open("data/train.csv", "r").readlines()
shit = open('train_submission.csv', "r").readlines()
diff = open("difference.csv", "x")
diff.write("image,label")
for idx in range(len(train)):
    if train[idx] != shit[idx]:
        diff.write(train[idx]) 
