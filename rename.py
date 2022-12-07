import os
t1='Dataset/t1v2'
t2='Dataset/t2v2'

for i,(j,k) in enumerate(zip(os.listdir(t1),os.listdir(t2))):
    t1p = os.path.join(t1,j)
    t2p = os.path.join(t2,k)
    os.rename(t1p,os.path.join(t1,'augv2_'+str(i)+'.jpg'))
    os.rename(t2p,os.path.join(t2,'augv2_'+str(i)+'.jpg'))
    