from datasets import load_dataset

dataset = load_dataset("Bingsu/Human_Action_Recognition")
count = 0
for img in dataset["train"]:
    if img['labels'] == 11:
        count += 1
        img['image'].save(f"./trainSet/{count}.jpg")

print(f"train set count: {count}")
count = 0
for img in dataset["test"]:
    if img['labels'] == 11:
        count += 1

print(f"test set count: {count}")