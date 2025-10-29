from synprotac.chemistry import visualize_path
import pickle 
import os 
from pathlib import Path

for j in range(2,3):
    #try:
        data_path=Path(f"./rl-samples/train/{j}")
        with open(data_path/f"routes-0.pkl", 'rb') as f:
            routes = pickle.load(f)[0]
        savepath= data_path/"pics"
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        print ('*****',len(routes))
        for i in range(len(routes)):
            print (i, routes[i])
            route_dict={
                "warhead":routes[i][0]["from_state"],
                "e3_ligand":routes[i][-1]["reagent"],
                "final_product":routes[i][-1]["product"],
                "reactions":routes[i]
            }
            visualize_path(route_dict, f"{savepath}/{i}.png")
    #except:
    #    pass
