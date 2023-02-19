import numpy as np

def bresenham(x1,y1,x2,y2):
    x1=np.round(x1)
    x2=np.round(x2)
    y1=np.round(y1)
    y2=np.round(y2)
    dx=np.abs(x2-x1)
    dy=np.abs(y2-y1)    
    dx2=np.floor(dx/2)
    step=-dy*dx+np.floor(dx/2)
    steep=np.abs(dy) > np.abs(dx)
    if steep:
        t=dx
        dx=dy
        dy=t
    if dy==0:
        # dx= np.ceil(dx)
        # dx = dx.astype(int)
        q=np.zeros((dx+1,1),dtype=int)
    else:
        Mod1 =np.arange(dx2,step,-dy)
        Mod1 = Mod1.reshape(np.size(Mod1),1)
        Remainder = np.remainder(Mod1,dx)
        Difference = Remainder[1:] - Remainder[:-1]
        Difference[Difference>=0]=1
        Difference[Difference<0]=0
        # Difference = Difference[Difference >= 0]
        zer = np.zeros((1,1))
        q = np.concatenate((zer,Difference), axis=0)
        # q=np.array([[0],[Difference]])
        # q=np.array([[0],[Difference]])
        
        
    if steep:
        if y1<=y2:
            y=np.arange(y1,y2,1).T
        else:
            y=np.arange(y1,y2,-1).T
        if x1<=x2:
            x=x1+np.cumsum(q)
        else:
            x=x1-np.cumsum(q)
    else:
        if x1<=x2:
            x=np.arange(x1,x2,1).T
        else:
            x=np.arange(x1,x2,-1).T
        if y1<=y2:
            y=y1+np.cumsum(q)
        else:
            y=y1-np.cumsum(q)
    x=x.astype(int)
    y=y.astype(int)
    return x, y
        
 











           
            
            
    
