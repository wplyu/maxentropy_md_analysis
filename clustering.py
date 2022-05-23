from base import *
def cal_rmsd_matrix(traj,nlen):
    import time
    import multiprocessing as mp
    import pandas as pd
    import mdtraj as mtj
    t0=time.time()
    mgr=mp.Manager()
    rmsd_matrix=mgr.dict()
    t0i=time.time()
    ttrunks=splitN(list(range(traj.n_frames)),nlen)
    print(len(ttrunks))
    bbidx=[i.index for i in traj.top.atoms if i.name in ['CA','CB']]
    def cal_rmsd(t,ttrunks,traj,rmsd_matrix,bbidx):
        import mdtraj as mtj
        t0i=time.time()
        for i in ttrunks[t]:
            traj=traj.superpose(traj,frame=i,atom_indices=bbidx)
            rmsd_matrix[i]=pd.Series(mtj.rmsd(traj[i:],traj[i],frame=0,atom_indices=bbidx,parallel=True),index=list(range(i,traj.n_frames)))
            if i%20==0 and i<=len(ttrunks[0]):
                print(i,round(time.time()-t0i,1))
                t0i=time.time()
    jobs={}
    for t in range(len(ttrunks)):
        jobs[t]=mp.Process(target=cal_rmsd,args=(t,ttrunks,traj,rmsd_matrix,bbidx))
        jobs[t].start()
    for t in range(len(ttrunks)):
        jobs[t].join()
    keys=sorted([i for i in rmsd_matrix.keys()])
    rmsd_matrix_s={}
    for i in sorted(keys):
        rmsd_matrix_s[i]=rmsd_matrix[i]
    rmsd2d=pd.DataFrame.from_dict(rmsd_matrix_s).fillna(0)
    rmsd2ds=rmsd2d+rmsd2d.T
    print(time.time()-t0)
    return rmsd2ds

def rmsd_matrix_clustering(xtrajf,xtopf,deltt):
    import mdtraj as mtj
    import gc
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    aporf_traj=mtj.load_xtc(xtrajf,top=xtopf)
    bbidx=[i.index for i in aporf_traj.top.atoms if i.name in ['CA','CB']]
    aporf_traj_bb=aporf_traj.atom_slice(bbidx)
    del aporf_traj
    gc.collect()
    print(aporf_traj_bb.n_atoms)
    nlen=200
    if aporf_traj_bb.n_frames==30000:
        framelist=list(range(0,aporf_traj_bb.n_frames,deltt))
    else:
        framelist=list(range(0,20000,deltt))
        _list=[framelist.append(f) for f in range(20000,21000,1)]
    framelist=list(range(0,aporf_traj_bb.n_frames,deltt))
    #print('n_conforms:',len(framelist))
    try:
        rmsd2d_aporf_1ns=pd.read_csv('/'.join(xtrajf.split('/')[:-1])+'/rmsd2d.csv',index_col=0)
        rmsd2d_aporf_1ns.columns=rmsd2d_aporf_1ns.columns.astype('int')
        print('rmsd2D has been loaded')
    except:
        rmsd2d_aporf_1ns=cal_rmsd_matrix(aporf_traj_bb[framelist],nlen)
        print('rmsd2D has been calculated')
        rmsd2d_aporf_1ns.to_csv('/'.join(xtrajf.split('/')[:-1])+'/rmsd2d.csv')
    rmsd2d_aporf_1ns_sim=1-rmsd2d_aporf_1ns/rmsd2d_aporf_1ns.max().max()
    #plt.imshow(rmsd2d_aporf_1ns_sim)
    #plt.colorbar()
    perf_range2=np.linspace(abs(rmsd2d_aporf_1ns_sim.values).min(),abs(rmsd2d_aporf_1ns_sim.values).max(),50)
    #print(perf_range2)
    aporf_rmsd_l2,aporf_ap_rms_l2=clustering_xyz_cov_mp(rmsd2d_aporf_1ns,[],perf_range2)
    aporf_rms_clustersize=[len([j for j in aporf_ap_rms_l2.labels_ if i==j]) for i in range(len(aporf_ap_rms_l2.cluster_centers_indices_))]
    aporf_ap_rms_l2.cluster_centers_indices_[np.array(aporf_rms_clustersize).argmax()]
    conidx=[framelist[i] for i in aporf_ap_rms_l2.cluster_centers_indices_]
    print('clustercenter:',conidx)
    print('clustersize:',aporf_rms_clustersize)
    print('Biggest cluster:',framelist[aporf_ap_rms_l2.cluster_centers_indices_[np.array(aporf_rms_clustersize).argmax()]])
    cls=pd.DataFrame(aporf_rms_clustersize,index=conidx)
    #plt.plot(cls)
    tfxx=xtrajf
    pfxx=xtopf
    try:
        _trajx=mtj.load_xtc(tfxx,top=pfxx)
    except:
        pass
    ii=0
    weigths={}
    for i in cls.sort_values(0,ascending=False).index:
        print(ii,i,cls.loc[i],'/'.join(tfxx.split('/')[:-1])+'/rep'+str(ii)+'.pdb')
        try:
            _trajx[i].save_pdb('/'.join(tfxx.split('/')[:-1])+'/rep'+str(ii)+'.pdb')
        except:
            pass
        ii+=1
        weigths[ii]=cls.loc[i]
    #print(weigths)
    (cls.sort_values(0,ascending=False)/cls.sum()).T.plot.bar(figsize=(10,5))
    plt.xlabel('Representatives',size=18)
    plt.ylabel('Weighting',size=18)
    plt.xlim(-0.3,0.3)
    plt.legend(fontsize='large')
    plt.yticks(size=18)
    plt.xticks(size=0)
    return cls#,rmsd2d_aporf_1ns
def clustering_xyz_cov_mp(similarities,ap_upper_level,perf_range):
    import mdtraj as mtj
    import gc
    import pandas as pd
    import numpy as np
    import multiprocessing as mp
    import matplotlib.pyplot as plt
    from sklearn.cluster import AffinityPropagation
    data_ap_xyz_pool={}
    ap_xyz_sum_pool={}
    #*similarities.mean()
    for s in similarities.index:
        similarities.loc[s][s]=0
    def run_ap_process(rid,q_dict,sum_dict):
        r=perf_range[rid]
        perfe=pd.DataFrame(r*np.ones(similarities.shape[0]),index=similarities.index)
        perfm=r*similarities.mean()
        seed = np.random.RandomState(seed=9983)
        try:
            ap_oute=AffinityPropagation(affinity='precomputed',max_iter=500,preference=perfe,random_state=seed).fit(similarities)
            ap_outm=AffinityPropagation(affinity='precomputed',max_iter=500,preference=perfm,random_state=seed).fit(similarities)
            sum_oute=cal_ap_sum(ap_oute)
            sum_outm=cal_ap_sum(ap_outm)
            if sum_oute>sum_outm:
                data_ap_xyz=ap_oute
                ap_xyz_sum=sum_oute
                eperf=True
            else:
                data_ap_xyz=ap_outm
                ap_xyz_sum=sum_outm
                eperf=False
            q_dict[r]=data_ap_xyz
            sum_dict[r]=ap_xyz_sum
            print(rid,round(ap_xyz_sum,2),len(data_ap_xyz.cluster_centers_indices_),flush=True,end='')
        except(Exception) as error:
            print(rid,'cannot converge! It will be ignored!')
            print(error)
        return None
    data_ap_xyz_pool=mp.Manager().dict()
    ap_xyz_sum_pool=mp.Manager().dict()
    jobs={}
    for rid in range(len(perf_range)):
        jobs[rid]=mp.Process(target=run_ap_process,args=(rid,data_ap_xyz_pool,ap_xyz_sum_pool))
        jobs[rid].start()
    for j in jobs.keys():
        jobs[j].join()
    pd.DataFrame.from_dict(ap_xyz_sum_pool,orient='index').sort_index().plot(style='-*')
    plt.show()
    rmax=pd.DataFrame.from_dict(ap_xyz_sum_pool,orient='index').idxmax().values[0]
    data_ap_xyz=data_ap_xyz_pool[rmax]
    print('R_max=',rmax,'Sum_max=',ap_xyz_sum_pool[rmax])
    print(len(data_ap_xyz.cluster_centers_indices_))
    print(1+data_ap_xyz.cluster_centers_indices_)
    #plt.imshow(similarities.iloc[data_ap_xyz.cluster_centers_indices_,data_ap_xyz.cluster_centers_indices_])
    #plt.colorbar()
    #plt.show()
    print(similarities.iloc[data_ap_xyz.cluster_centers_indices_,data_ap_xyz.cluster_centers_indices_].median().median())
    similarities_l2=pd.DataFrame(index=similarities.iloc[data_ap_xyz.cluster_centers_indices_].index,columns=similarities.iloc[data_ap_xyz.cluster_centers_indices_].index)
    for x in range(len(data_ap_xyz.cluster_centers_indices_)):
        cx=[similarities.index[i] for i in range(similarities.shape[0]) if data_ap_xyz.labels_[i]==x]
        for y in range(len(data_ap_xyz.cluster_centers_indices_)):
            cy=[similarities.index[i] for i in range(similarities.shape[0]) if data_ap_xyz.labels_[i]==y]
            #print(similarities.index[x],similarities.index[y])
            #print(x,y,cx,cy)
            similarities_l2.at[similarities.index[data_ap_xyz.cluster_centers_indices_[x]],\
                               similarities.index[data_ap_xyz.cluster_centers_indices_[y]]]=\
                                similarities.loc[cx][cy].mean().mean()
    #plt.imshow(similarities_l2.astype('float32'))
    #plt.colorbar()
    #plt.show()
    return similarities_l2.astype('float32'),data_ap_xyz

def cal_ap_sum(ap_xyz_l2):
    import pandas as pd
    c_id={}
    labels_kh=pd.DataFrame(ap_xyz_l2.labels_,index=list(range(len(ap_xyz_l2.labels_))),columns=['label'])
    for c in range(len(ap_xyz_l2.cluster_centers_indices_)):
        ci=ap_xyz_l2.cluster_centers_indices_[c]
        temp=labels_kh[labels_kh==c].dropna()
        c_id[ci]=ap_xyz_l2.affinity_matrix_[ci][temp.index].sum()
    c_id=pd.DataFrame.from_dict(c_id,orient='index')
    c_id_sum=c_id.sum().values[0]
    return c_id_sum