import random
from randomSet import seed_everything

seed_everything(42)
data_list = [1, 2, 3, 4, 5]
shuffled_list = data_list[:]
print(shuffled_list)
random.shuffle(shuffled_list)
print(shuffled_list)