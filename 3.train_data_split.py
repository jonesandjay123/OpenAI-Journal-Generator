from sklearn.model_selection import train_test_split

with open("train_data.txt", "r") as f:
    data = f.readlines()

train_data, val_data = train_test_split(data, test_size=0.1)

with open("train.txt", "w") as f:
    f.writelines(train_data)

with open("val.txt", "w") as f:
    f.writelines(val_data)
