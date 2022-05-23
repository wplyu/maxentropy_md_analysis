def splitN(slist,size):
    out=[]
    N=int(len(slist)/size)
    for i in range(N):
        out.append(slist[i*size:(i+1)*size])
    if len(slist)%size!=0:
        out.append(slist[N*size:])
    return out

def split(slist,N):
    out=[]
    size=int(len(slist)/N)
    for i in range(N-1):
        out.append(slist[i*size:(i+1)*size])
    out.append(slist[(N-1)*size:])
    return out

def load_index(idxfile):
    outlist={}
    data=open(idxfile).readlines()
    for i in range(len(data)):
        line=[l for l in data[i][:-1].split(',') if l!=''][1:]
        #print(line)
        id0=set([l.split('_')[0] for l in line if l.split('_')[0]!=''])
        id1=set([l.split('_')[1] for l in line if l.split('_')[1]!=''])
        id12='_'+'_'.join(id0)+'_@_'+'_'.join(id1)+'_#'+str(i)
        outlist[id12]=line
    idx0=[i.split('#')[0].split('@')[0] for i in outlist.keys()]
    idx1=[i.split('#')[0].split('@')[1] for i in outlist.keys()]
    fidx=[i.split('#')[1] for i in outlist.keys()]
    return outlist,[idx0,idx1,fidx]


