from synprotac.chemistry import visualize_path
import pickle 
import os 
from pathlib import Path
with open("routes.pkl", 'rb') as f:
    data = pickle.load(f)
savepath="pics"
if not os.path.exists(savepath):
    os.makedirs(savepath)
for i in range(len(data)):
    visualize_path(data[i], f"{savepath}/{i}.png")