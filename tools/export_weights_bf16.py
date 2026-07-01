"""Export all backbone weights + ref as raw f32 binary for the C++ host."""
import numpy as np, os
D="/home/amd/project/npu_iron/weights/vit_full"; OUT="/home/amd/project/npu_iron/weights/cbb"; os.makedirs(OUT,exist_ok=True)
C=1024; Hid=4736; Hpad=5120
def L(n): return np.load(f"{D}/{n}.npy")
def save(name,arr): arr.astype(np.float32).tofile(f"{OUT}/{name}.bin")
for i in range(32):
    qw=L(f"L{i}_qw"); kw=L(f"L{i}_kw"); vw=L(f"L{i}_vw")
    Wqkv=np.concatenate([qw.T,kw.T,vw.T],axis=1)          # [1024,3072]
    bqkv=np.concatenate([L(f"L{i}_qb"),L(f"L{i}_kb"),L(f"L{i}_vb")])  # [3072]
    save(f"L{i}_Wqkv",Wqkv); save(f"L{i}_bqkv",bqkv)
    save(f"L{i}_Ow",L(f"L{i}_ow").T); save(f"L{i}_Ob",L(f"L{i}_ob"))
    save(f"L{i}_ln1w",L(f"L{i}_ln1w")); save(f"L{i}_ln1b",L(f"L{i}_ln1b"))
    save(f"L{i}_ln2w",L(f"L{i}_ln2w")); save(f"L{i}_ln2b",L(f"L{i}_ln2b"))
    W1=np.zeros((C,Hpad),np.float32); W1[:,:Hid]=L(f"L{i}_fc1w").T; save(f"L{i}_W1",W1)
    b1=np.zeros(Hpad,np.float32); b1[:Hid]=L(f"L{i}_fc1b"); save(f"L{i}_b1",b1)
    W2=np.zeros((Hpad,C),np.float32); W2[:Hid]=L(f"L{i}_fc2w").T; save(f"L{i}_W2",W2)
    save(f"L{i}_fc2b",L(f"L{i}_fc2b"))
save("rope_win_cos",L("rope_win_cos")); save("rope_win_sin",L("rope_win_sin"))
save("rope_glob_cos",L("rope_glob_cos")); save("rope_glob_sin",L("rope_glob_sin"))
save("block0_in",L("block0_in").reshape(1296,C)); save("final_feat",L("final_feat").reshape(1296,C))
print("exported to",OUT, "files:",len(os.listdir(OUT)))
