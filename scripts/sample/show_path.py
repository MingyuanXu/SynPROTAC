from synprotac.chemistry import visualize_path
import pickle 
import os 
from pathlib import Path

for j in range(10):
    try:
        data_path=Path(f"./samples/{j}")
        with open(data_path/f"routes-{j}.pkl", 'rb') as f:
            routes = pickle.load(f)
        savepath= data_path/"pics"
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        print (len(routes))
        for i in range(len(routes)):
            print (i, routes[i])
            route_dict={
                "warhead":routes[i][0]["from_state"],
                "e3_ligand":routes[i][-1]["reagent"],
                "final_product":routes[i][-1]["product"],
                "reactions":routes[i]
            }
            visualize_path(route_dict, f"{savepath}/{i}.png")
    except:
        pass
