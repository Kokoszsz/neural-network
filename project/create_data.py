import random

data = {2: 0, 1: 1, 0: 0}

with open("data.txt", "w") as file:
    for _ in range(100):
        key, value = random.choice(list(data.items()))
        file.write(f"{key},{value}\n")
