# SynPROTAC
Official code of "Design synthesizable PROTACs through synthesis constrained generative model and reinforcement learning"

Protein hydrolysis targeting chimeric (PROTAC) has emerged as a promising technology in degrading disease-related proteins for drug design. Recent deep generative models can accelerate PROTAC design, but the generated molecules are often difficult to synthesize. Here we develop SynPROTAC model, which integrates chemical reaction path driven molecule assembly with reinforcement learning for design of synthesizable PROTACs together with favorable binding properties. Specifically, the synthesis constrained generative model employs Graphormer encoded warhead or E3 ligand as input, and autoregressively samples reaction templates and building blocks through transformer based decoder and chemical fingerprint based searching along with transfer learning for PROTAC construction. The comprehensive evaluations indicated that SynPROTAC is capable of generating new PROTACs with feasible synthetic routes, reasonable physico-chemical and binding related properties. We further applied SynPROTAC to design PROTAC molecules degrading bromodomain-containing protein 4 (BRD4), and two selected compounds were successfully synthesized according to the synthetic routes proposed by SynPROTAC. In the following biological experiments, both of them exhibited nanomolar-level degradation activity against BRD4 and potent anti-proliferation activity against MV411 tumor cell. These results demonstrate the capability of SynPROTAC to design novel bioactive PROTAC molecules with feasible synthetic routes. 

## Installations
Environment install:

	conda env create -f environment.yaml 


SynPROTAC installation:

	cd synprotac_project

	pip install -e .

Training:

	cd scripts/train

	python train.py -i ctrl.json 

Here, we only provide a simple toy dataset to test the training workflow. The whole 20 millions of synthesizable routes is about 80G in cloud, it can be download from https://figshare.com/articles/journal_contribution/The_pretrained_model_of_SynPROTAC_/30446639 soon. 

Sample:

	cd scripts/sample 

	python sample.py -i ctrl.json 

	python show_path.py 

The pretrained model are avaliable from  https://figshare.com/articles/journal_contribution/The_pretrained_model_of_SynPROTAC_/30446639. it should be moved to the ../../pretrained_models.  

Eval:

	cd scripts/eval 

	python eval.py -i ctrl.json 

Reinforcement Learning

	cd scripts/rl/2D_similarity or cd scripts/rl/Constrained_Docking

	python rl.py -i ctrl.json 

