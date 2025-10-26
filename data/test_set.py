with open("protac_warhead_linker_e3_ligand.csv","r") as f:
    lines=[line.strip() for line in f.readlines()]

with open("test.ids","r") as f:
    test_ids=[int(line.strip()) for line in f.readlines()]
f_test =open("testset.csv","w")
f_test.write("protac,warhead,linker_id,e3_ligand\n")
for lid,line in enumerate(lines):
    line=','.join(line.split(',')[1:])
    if lid in test_ids:
        f_test.write(line+"\n")
