from base import *
def traj_cov_cal(trajf,topf):
    import multiprocessing as mp
    import mdtraj as mtj
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    try:
        _cov_3d=pd.read_csv('.'.join(trajf.split('.')[:-1])+'_traj_cov_xyz.csv',index_col=0)
        _cov_3d.columns=_cov_3d.columns.astype('int')
        _xyz_cov={}
        for k in range(3):
            _xyz_cov[k]=pd.read_csv('.'.join(trajf.split('.')[:-1])+'_traj_cov_'+str(k)+'.csv',index_col=0)
            _xyz_cov[k].columns=_xyz_cov[k].columns.astype('int')
        plt.imshow(_cov_3d)
        plt.colorbar()
        plt.show()
        return _xyz_cov,_cov_3d,_cov_3d[_cov_3d>0],_cov_3d[_cov_3d<0]
    except:
        pass
    kh_traj=mtj.load_xtc(trajf,top=topf)
    kh_res=[[a.serial for a in r.atoms if a.element.atomic_number!=1] for r in kh_traj.top.residues]
    kh_mass=[[a.element.atomic_number for a in r.atoms if a.element.atomic_number!=1] for r in kh_traj.top.residues]
    kh_xyz_ca={}
    print(kh_res[0])
    for r in range(kh_traj.n_residues):
        #print(kh_res[r],kh_mass[r])
        _temp=np.asarray([kh_mass[r]]*3).T*kh_traj.xyz[:,kh_res[r]]
        #print(_temp.shape,kh_traj.xyz[:,kh_res[r]].shape)
        kh_xyz_ca[r+1]=_temp.mean(axis=1)
    print('COM of each residue has been calculated')
    kh_xyz_cov={}
    kh_xyz_cov[0]=pd.DataFrame(index=range(1,kh_traj.n_residues+1),columns=range(1,1+kh_traj.n_residues))
    kh_xyz_cov[1]=pd.DataFrame(index=range(1,kh_traj.n_residues+1),columns=range(1,1+kh_traj.n_residues))
    kh_xyz_cov[2]=pd.DataFrame(index=range(1,kh_traj.n_residues+1),columns=range(1,1+kh_traj.n_residues))
    for i in range(kh_traj.n_residues):
        for j in range(i+1,kh_traj.n_residues):
            for z in range(3):
                covijz=np.cov(kh_xyz_ca[i+1][:,z],kh_xyz_ca[j+1][:,z])
                kh_xyz_cov[z].at[i+1,j+1]=covijz[0,1]/(covijz[0,0]*covijz[1,1])**0.5
                kh_xyz_cov[z].at[j+1,i+1]=kh_xyz_cov[z].at[i+1,j+1]
    print('COV matrix of each dimension has been calculated')
    for k in kh_xyz_cov.keys():
        kh_xyz_cov[k].to_csv('.'.join(trajf.split('.')[:-1])+'_traj_cov_'+str(k)+'.csv')
    kh_cov_x=kh_xyz_cov[0].fillna(0)
    kh_cov_y=kh_xyz_cov[1].fillna(0)
    kh_cov_z=kh_xyz_cov[2].fillna(0)
    kh_cov_x1=kh_cov_x[kh_cov_x<=0].fillna(1)
    kh_cov_xs=kh_cov_x1[kh_cov_x1>=0].fillna(-1)
    kh_cov_y1=kh_cov_y[kh_cov_y<=0].fillna(1)
    kh_cov_ys=kh_cov_y1[kh_cov_y1>=0].fillna(-1)
    kh_cov_z1=kh_cov_z[kh_cov_z<=0].fillna(1)
    kh_cov_zs=kh_cov_z1[kh_cov_z1>=0].fillna(-1)
    kh_cov_xyzp=(kh_cov_xs*kh_cov_x**2+kh_cov_ys*kh_cov_y**2+kh_cov_zs*kh_cov_z**2)**0.5
    kh_cov_xyzn=-1*(-1*(kh_cov_xs*kh_cov_x**2+kh_cov_ys*kh_cov_y**2+kh_cov_zs*kh_cov_z**2))**0.5
    kh_cov_3d=kh_cov_xyzp.fillna(0)+kh_cov_xyzn.fillna(0)
    kh_cov_3d.to_csv('.'.join(trajf.split('.')[:-1])+'_traj_cov_xyz.csv')
    plt.imshow(kh_cov_3d)
    plt.colorbar()
    plt.show()
    return kh_xyz_cov,kh_cov_3d,kh_cov_xyzp,kh_cov_xyzn

def traj_cov_cal_mp(trajf,topf):
    import multiprocessing as mp
    import mdtraj as mtj
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    try:
        _cov_3d=pd.read_csv('.'.join(trajf.split('.')[:-1])+'_traj_cov_xyz.csv',index_col=0)
        _cov_3d.columns=_cov_3d.columns.astype('int')
        _xyz_cov={}
        for k in range(3):
            _xyz_cov[k]=pd.read_csv('.'.join(trajf.split('.')[:-1])+'_traj_cov_'+str(k)+'.csv',index_col=0)
            _xyz_cov[k].columns=_xyz_cov[k].columns.astype('int')
        plt.imshow(_cov_3d)
        plt.colorbar()
        plt.show()
        return _xyz_cov,_cov_3d,_cov_3d[_cov_3d>0],_cov_3d[_cov_3d<0]
    except:
        pass
    _traj=mtj.load_xtc(trajf,top=topf)
    _res=[[a.serial for a in r.atoms if a.element.atomic_number!=1] for r in _traj.top.residues]
    _mass=[[a.element.atomic_number for a in r.atoms if a.element.atomic_number!=1] for r in _traj.top.residues]
    _xyz_ca={}
    print(_res[0])
    for r in range(_traj.n_residues):
        #print(kh_res[r],kh_mass[r])
        _temp=np.asarray([_mass[r]]*3).T*_traj.xyz[:,_res[r]]
        _xyz_ca[r+1]=_temp.mean(axis=1)
    print('COM of each residue has been calculated')
    _xyz_cov={}
    #_xyz_cov[0]=pd.DataFrame(index=range(1,_traj.n_residues+1),columns=range(1,1+_traj.n_residues))
    #_xyz_cov[1]=pd.DataFrame(index=range(1,_traj.n_residues+1),columns=range(1,1+_traj.n_residues))
    #_xyz_cov[2]=pd.DataFrame(index=range(1,_traj.n_residues+1),columns=range(1,1+_traj.n_residues))
    n_reschunk=splitN(range(_traj.n_residues),40)
    print('Chunks:',len(n_reschunk))
    def cal_cov_res(res_list,q_out):
        _xyz_cov={}
        _xyz_cov[0]=pd.DataFrame(index=range(1,_traj.n_residues+1),columns=range(1,1+_traj.n_residues))
        _xyz_cov[1]=pd.DataFrame(index=range(1,_traj.n_residues+1),columns=range(1,1+_traj.n_residues))
        _xyz_cov[2]=pd.DataFrame(index=range(1,_traj.n_residues+1),columns=range(1,1+_traj.n_residues))
        for i in res_list:
            for j in range(i+1,_traj.n_residues):
                for z in range(3):
                    covijz=np.cov(_xyz_ca[i+1][:,z],_xyz_ca[j+1][:,z])
                    _xyz_cov[z].at[i+1,j+1]=covijz[0,1]/(covijz[0,0]*covijz[1,1])**0.5
                    _xyz_cov[z].at[j+1,i+1]=_xyz_cov[z].at[i+1,j+1]
        q_out.put(_xyz_cov)
    q_out=mp.Queue()
    jobs={}
    for j in range(len(n_reschunk)):
        jobs[j]=mp.Process(target=cal_cov_res,args=(n_reschunk[j],q_out))
        jobs[j].start()
    for j in jobs.keys():
        jobs[j].join()
    #_out_list=[]
    _xyz_covx={}
    _xyz_covx[0]=[]
    _xyz_covx[1]=[]
    _xyz_covx[2]=[]
    for j in jobs:
        _temp=q_out.get()
        _xyz_covx[0].append(_temp[0])
        _xyz_covx[1].append(_temp[1])
        _xyz_covx[2].append(_temp[2])
    _xyz_cov[0]=_xyz_covx[0][0]
    _xyz_cov[1]=_xyz_covx[1][0]
    _xyz_cov[2]=_xyz_covx[2][0]    
    for i in range(len(n_reschunk)):
        _xyz_cov[0]+=_xyz_covx[0][i]
        _xyz_cov[1]+=_xyz_covx[1][i]
        _xyz_cov[2]+=_xyz_covx[2][i]
    print('COV matrix of each dimension has been calculated')
    for k in _xyz_cov.keys():
        _xyz_cov[k].to_csv('.'.join(trajf.split('.')[:-1])+'_traj_cov_'+str(k)+'.csv')
    _cov_x=_xyz_cov[0].fillna(0)
    _cov_y=_xyz_cov[1].fillna(0)
    _cov_z=_xyz_cov[2].fillna(0)
    _cov_x1=_cov_x[_cov_x<=0].fillna(1)
    _cov_xs=_cov_x1[_cov_x1>=0].fillna(-1)
    _cov_y1=_cov_y[_cov_y<=0].fillna(1)
    _cov_ys=_cov_y1[_cov_y1>=0].fillna(-1)
    _cov_z1=_cov_z[_cov_z<=0].fillna(1)
    _cov_zs=_cov_z1[_cov_z1>=0].fillna(-1)
    _cov_xyzp=(_cov_xs*_cov_x**2+_cov_ys*_cov_y**2+_cov_zs*_cov_z**2)**0.5
    _cov_xyzn=-1*(-1*(_cov_xs*_cov_x**2+_cov_ys*_cov_y**2+_cov_zs*_cov_z**2))**0.5
    _cov_3d=_cov_xyzp.fillna(0)+_cov_xyzn.fillna(0)
    _cov_3d.to_csv('.'.join(trajf.split('.')[:-1])+'_traj_cov_xyz.csv')
    plt.imshow(_cov_3d)
    plt.colorbar()
    plt.show()
    return _xyz_cov,_cov_3d,_cov_xyzp,_cov_xyzn

def ap_search(data_ap_xyz_pool,ap_xyz_sum_pool,perf_list,perf_range,similarities):
    import mdtraj as mtj
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import AffinityPropagation
    for rid in perf_list:
        r=perf_range[rid]
        #print(perfe.sum(),perfm.sum())
        try:
            perfe=pd.DataFrame(r*np.ones(similarities.shape[0]),index=similarities.index)
            perfm=r*similarities.mean()
            seed = np.random.RandomState(seed=9983)
            ap_oute=AffinityPropagation(affinity='precomputed',max_iter=1500,preference=perfe,random_state=seed).fit(similarities)
            ap_outm=AffinityPropagation(affinity='precomputed',max_iter=1500,preference=perfm,random_state=seed).fit(similarities)
        except:
            perfe=pd.DataFrame(0.0001+r*np.ones(similarities.shape[0]),index=similarities.index)
            perfm=0.0001+r*similarities.mean()
            seed = np.random.RandomState(seed=9983)
            ap_oute=AffinityPropagation(affinity='precomputed',max_iter=1500,preference=perfe,random_state=seed).fit(similarities)
            ap_outm=AffinityPropagation(affinity='precomputed',max_iter=1500,preference=perfm,random_state=seed).fit(similarities)
        try:
            if cal_ap_sum(ap_oute)>cal_ap_sum(ap_outm):
                data_ap_xyz_pool[r]=ap_oute
                ap_xyz_sum_pool[r]=cal_ap_sum(ap_oute)
                eperf=True
                #print('E_perf',ap_xyz_sum_pool[r])
            else:
                data_ap_xyz_pool[r]=ap_outm
                ap_xyz_sum_pool[r]=cal_ap_sum(ap_outm)
                eperf=False
        except:
            pass
    print('job is done',perf_list[0],perf_list[-1])
        
def ap_searching_mp(N,perf_range,similarities):
    import multiprocessing as mp
    mgr=mp.Manager()
    mgr_ap_xyz=mgr.dict()
    mgr_ap_sum=mgr.dict()
    perf_lists=splitN(range(len(perf_range)),N)
    print('AP_searching by n_jobs:',len(perf_lists))
    jobs={}
    for pix in range(len(perf_lists)):
        jobs[pix]=mp.Process(target=ap_search,args=(mgr_ap_xyz,mgr_ap_sum,perf_lists[pix],perf_range,similarities))
        jobs[pix].start()
        #print('Starting job',pix)
    for pix in range(len(perf_lists)):
        jobs[pix].join()
        #print('Finishing',pix)
        #print(rid,round(ap_xyz_sum_pool[r],2),'Max',round(max([ap_xyz_sum_pool[x] for x in ap_xyz_sum_pool.keys()]),2),len(data_ap_xyz_pool[r].cluster_centers_indices_),flush=True,end='')
    return mgr_ap_xyz,mgr_ap_sum

def clustering_xyz_cov(similarities,ap_upper_level,perf_range,pbool,zbool,fname):
    import multiprocessing as mp
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import manifold
    from sklearn.decomposition import PCA
    import matplotlib.colors as colors
    import matplotlib.cm as cm
    #*similarities.mean()
    pml_list={}
    print('Resid:',similarities.index)
    similarities.columns=similarities.columns.astype('int64')
    for s in similarities.index:
        similarities.loc[s][s]=0
    mgr_ap_xyz,mgr_ap_sum=ap_searching_mp(5,perf_range,similarities)
    #print(mgr_ap_xyz,mgr_ap_sum)
    pd.DataFrame.from_dict(mgr_ap_sum,orient='index').plot(style='*')
    plt.show()
    rmax=pd.DataFrame.from_dict(mgr_ap_sum,orient='index').idxmax().values[0]
    data_ap_xyz=mgr_ap_xyz[rmax]
    print('R_max=',rmax,'Sum_max=',mgr_ap_sum[rmax])
    print(len(data_ap_xyz.cluster_centers_indices_))
    print(1+data_ap_xyz.cluster_centers_indices_)
    plt.imshow(similarities.iloc[data_ap_xyz.cluster_centers_indices_,data_ap_xyz.cluster_centers_indices_])
    plt.colorbar()
    plt.show()
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
    plt.imshow(similarities_l2.astype('float32'))
    plt.colorbar()
    plt.show()
    if pbool==False:
        return similarities_l2.astype('float32'),data_ap_xyz
    seed = np.random.RandomState(seed=3)
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                       dissimilarity="precomputed", n_jobs=1).fit(1/(1+similarities/(similarities.max().max()-similarities.min().min()))**2)
    pos = mds.embedding_
    print('mds is ok')
    nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
                        dissimilarity="precomputed", random_state=seed, n_jobs=1,
                        n_init=1).fit_transform(1/(1+similarities/(similarities.max().max()-similarities.min().min()))**2, init=pos)
    print('nmds is ok')
    npos = nmds
    clf = PCA(n_components=2)
    tnpos = clf.fit_transform(npos)
    tnpos[:,1]*=zbool
    print('pca is ok')
    edge_cut=0.5
    s = 20
    cnorm = colors.Normalize(0,len(data_ap_xyz.cluster_centers_indices_))
    pycolor = cm.jet(cnorm(list(range(len(data_ap_xyz.cluster_centers_indices_)))))
    pymol_color='red,raspberry,darksalmon,brown,salmon,warmpink,ruby,green,chartreuse,splitpea,smudge,palegreen,\
                lime,limon,forest,blue,tv_blue,marine,slate,lightblue,skyblue,purplebule,deepblue,density,\
                yellow,paleyellow,yelloworange,wheat,sand,magenta,lightmagenta,hotpink,pink,lightpink,violet,\
                violetpurple,purple,deeppurple,cyan,palecyan,aquamarine,greencyan,teal,deepteal,lighteal,\
                orange,brightorange,lightorange,yelloworange,olive,deepolive,wheat,palegreen,lightblue,\
                paleyellow,palecyan,bluewhite,white,gray90,gray80,gray70,gray60,gray50,gray40,gray30,gray20,gray10,black'.split(',')
    pymol_color=list(set(pymol_color))
    plt.imshow(pycolor.T)
    plt.show()
    fig = plt.figure(1,figsize=(6,6))
    ax = plt.axes([0., 0., 1., 1.])
    print(len(data_ap_xyz.cluster_centers_indices_))
    #print('Centers:',data_ap_xyz.cluster_centers_indices_)
    tot_res=0
    print('bg_color white')
    csize=pd.DataFrame(index=range(len(data_ap_xyz.cluster_centers_indices_)))
    for k in range(len(data_ap_xyz.cluster_centers_indices_)):
        if len(ap_upper_level)!=0:
            cluster0=[i for i in range(len(data_ap_xyz.labels_)) if data_ap_xyz.labels_[i]==k]
            #print(cluster0,[similarities.index[i] for i in cluster0])
            cluster=[ap_upper_level.index[j]-1 for j in range(len(ap_upper_level)) if ap_upper_level.iloc[j]['L2'] in cluster0]
            #print(cluster)
        else:
            cluster0=[i for i in range(len(data_ap_xyz.labels_)) if data_ap_xyz.labels_[i]==k]
            cluster=[similarities.index[i]-1 for i in range(len(data_ap_xyz.labels_)) if data_ap_xyz.labels_[i]==k]
        tot_res=tot_res+len(cluster)
        print('select C'+str(k+1)+', resid ','+'.join(np.asarray(cluster).astype('str')))
        pml_list[k]=np.asarray(cluster)
        print('color '+pymol_color[k]+', C'+str(k+1))
        plt.plot(tnpos.T[0][cluster0].T, tnpos.T[1][cluster0].T,'.',color=pycolor[k],markersize=5)
        #plt.colorbar()
        #'''
        for j in range(k+1,len(data_ap_xyz.cluster_centers_indices_)):
            if similarities_l2.iat[k,j]>edge_cut:
                colx='grey'
                stylex='-'
            elif similarities_l2.iat[k,j]<-edge_cut:
                colx='grey'
                stylex='dashed'
            if similarities_l2.iat[k,j]>edge_cut:
                #print(k,j)
                poskj=tnpos[[data_ap_xyz.cluster_centers_indices_[k],data_ap_xyz.cluster_centers_indices_[j]]]
                plt.plot(poskj.T[0],poskj.T[1],colx,linestyle=stylex,linewidth=0.5*abs(similarities_l2.iat[k,j])**2)
        #print('Center:',k,data_ap_xyz.cluster_centers_indices_[k],tnpos.T[0][data_ap_xyz.cluster_centers_indices_[k]], tnpos.T[1][[data_ap_xyz.cluster_centers_indices_[k]]],1+similarities.index[data_ap_xyz.cluster_centers_indices_[k]])
        plt.text(tnpos.T[0][data_ap_xyz.cluster_centers_indices_[k]], tnpos.T[1][[data_ap_xyz.cluster_centers_indices_[k]]],similarities.index[data_ap_xyz.cluster_centers_indices_[k]],size=10)
        plt.plot(tnpos.T[0][data_ap_xyz.cluster_centers_indices_[k]], tnpos.T[1][[data_ap_xyz.cluster_centers_indices_[k]]],'*',color=pycolor[k],markersize=10)
        #print(similarities.index[k])
        #'''
    if len(ap_upper_level)!=0:
        fname+='/synconnet_L3.png'
    else:
        fname+='/synconnet_L2.png'
        for xxx in [450,484]:
            plt.text(tnpos.T[0][xxx],tnpos.T[1][xxx],1+xxx,size=10,color='red')
 
    plt.savefig(fname,dpi=300,transparent=True)
    plt.show()
    print(tot_res)
    return similarities_l2.astype('float32'),data_ap_xyz,tnpos,pml_list

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