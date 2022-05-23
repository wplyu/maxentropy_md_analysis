from base import *
def cov_pathway(cov_3d,cutoff):
    import networkx as nx
    g1=nx.Graph()
    g1.add_nodes_from(cov_3d.index-1)
    for i in cov_3d.index-1:
        for j in cov_3d.index-1:
            if i<=j:
                continue
            elif cov_3d.iat[i,j]>cutoff:
                g1.add_edge(i,j,weight=cov_3d.iat[i,j],capacity=cov_3d.iat[i,j])
    return g1
def residual_network_domain(similarities,dist_av_std,rs,rt,zbool,figname):
    import pandas as pd
    import numpy as np
    import multiprocessing as mp
    from sklearn.cluster import AffinityPropagation
    import matplotlib.pyplot as plt
    from sklearn import cluster, covariance, manifold
    from sklearn.decomposition import PCA
    from matplotlib.collections import LineCollection
    import networkx as nx
    from networkx.algorithms.flow import shortest_augmenting_path
    import matplotlib.colors as colors
    import matplotlib.cm as cm
    domains={'Nterm':list(range(0,131))+list(range(337,370)),'Exon':list(range(131,337)),\
             'Palm':list(range(370,446))+list(range(498,606)),'Finger':list(range(446,498)),\
             'Thumb':list(range(606,756)),'Pdna':list(range(756,768)),'Tdna':list(range(768,782))}
    dom_colors={'Nterm':'blue','Exon':'olive','Palm':'cyan','Finger':'orange','Thumb':'green','Pdna':'purple','Tdna':'brown'}
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
    pymol_color='red,raspberry,darksalmon,brown,salmon,warmpink,ruby,green,chartreuse,splitpea,smudge,palegreen,\
                lime,limon,forest,blue,tv_blue,marine,slate,lightblue,skyblue,purplebule,deepblue,density,\
                yellow,paleyellow,yelloworange,wheat,sand,magenta,lightmagenta,hotpink,pink,lightpink,violet,\
                violetpurple,purple,deeppurple,cyan,palecyan,aquamarine,greencyan,teal,deepteal,lighteal,\
                orange,brightorange,lightorange,yelloworange,olive,deepolive,wheat,palegreen,lightblue,\
                paleyellow,palecyan,bluewhite,white,gray90,gray80,gray70,gray60,gray50,gray40,gray30,gray20,gray10,black'.split(',')

    #'''
    G1=cov_pathway(similarities,0)
    mgr=mp.Manager()
    maxf_nets=mgr.dict()
    maxf_netsv=mgr.dict()
    def get_resnet(G1,s,t,nets,netsv):
        net_temp=nx.flow.maxflow.maximum_flow(G1,s,t,capacity='capacity')
        nets[str(s)+'_'+str(t)]=pd.concat([pd.DataFrame(net_temp[1][i],index=[i]) for i in net_temp[1].keys()]).fillna(0)
        netsv[str(s)+'_'+str(t)]=net_temp[0]
        #print(s,t,'is done')
    jobs={}
    for i in range(len(rs)):
        for j in range(len(rt)):
            keyij=str(rs[i]+1)+'_'+str(rt[j]+1)
            jobs[keyij]=mp.Process(target=get_resnet,args=(G1,rs[i],rt[j],maxf_nets,maxf_netsv))
            jobs[keyij].start()
        for j in range(len(rt)):
            keyij=str(rs[i]+1)+'_'+str(rt[j]+1)
            jobs[keyij].join()
    keyijs=[k for k in maxf_nets.keys()]
    print(keyijs)
    maxf_resnet_pd=pd.DataFrame(np.zeros(similarities.shape),index=G1.nodes,columns=G1.nodes)
    maxf_netsvalue={}
    for k in maxf_nets.keys():
        maxf_resnet_pd+=maxf_nets[k]
        maxf_netsvalue[k]=maxf_netsv[k]
    maxf_resnet_pd/=len(maxf_nets)
    plt.imshow(maxf_resnet_pd)
    plt.colorbar()
    plt.show()
    fig = plt.figure(1,figsize=(12,12))
    ax = plt.axes([0., 0., 1., 1.])
    lines_r=[]
    lines_g=[]
    for i in range(maxf_resnet_pd.shape[0]):
        if i%100==0:
            print('running:',i)
        #if i>40:
        #    continue
        for j in range(maxf_resnet_pd.shape[1]):
            if maxf_resnet_pd.iat[i,j]<=0.25*edge_cut:
                continue
            if i!=j and maxf_resnet_pd.iat[i,j]<edge_cut:
                deltxy=tnpos[j]-tnpos[i]
                plt.arrow(tnpos[i][0],tnpos[i][1],deltxy[0],deltxy[1],head_width=8*0.0025*maxf_resnet_pd.iat[i,j],\
                          head_length=8*0.0025*maxf_resnet_pd.iat[i,j],linewidth=maxf_resnet_pd.iat[i,j],\
                          length_includes_head=True,color='grey',alpha=0.3)
                lines_r.append([tnpos.T[0][[i,j]].T,tnpos.T[1][[i,j]].T])
        for j in range(maxf_resnet_pd.shape[1]):
            if maxf_resnet_pd.iat[i,j]<=0.25*edge_cut:
                continue
            if i!=j and maxf_resnet_pd.iat[i,j]>=edge_cut:
                #plt.plot(tnpos.T[0][[i,j]].T,tnpos.T[1][[i,j]].T,linestyle='-',color='red',linewidth=maxf_resnet_pd.iat[i,j])
                deltxy=tnpos[j]-tnpos[i]
                plt.arrow(tnpos[i][0],tnpos[i][1],deltxy[0],deltxy[1],head_width=8*0.0025*maxf_resnet_pd.iat[i,j],\
                          head_length=8*0.0025*maxf_resnet_pd.iat[i,j],linewidth=maxf_resnet_pd.iat[i,j],\
                          length_includes_head=True,color='red',alpha=0.4)
                lines_r.append([tnpos.T[0][[i,j]].T,tnpos.T[1][[i,j]].T])
    #'''
    for k in domains.keys():
        plt.plot(tnpos.T[0][domains[k]].T, tnpos.T[1][domains[k]].T,'.',color=dom_colors[k],markersize=15)
        #plt.plot(tnpos.T[0][data_ap_xyz.cluster_centers_indices_[k]], tnpos.T[1][[data_ap_xyz.cluster_centers_indices_[k]]],'*',color=pycolor[k],markersize=10)
        #plt.text(tnpos.T[0][data_ap_xyz.cluster_centers_indices_[k]], tnpos.T[1][[data_ap_xyz.cluster_centers_indices_[k]]],similarities.index[data_ap_xyz.cluster_centers_indices_[k]],size=8,alpha=0.8)
    plt.plot(tnpos.T[0][rs].T, tnpos.T[1][rs].T,'o',color='red',alpha=0.6,markersize=20)
    #plt.plot(tnpos.T[0][list(range(756,782))].T, tnpos.T[1][list(range(756,782))].T,'s',color='grey',alpha=0.3,markersize=20)
    plt.plot(tnpos.T[0][rt].T, tnpos.T[1][rt].T,'o',color='white',alpha=0.5,markersize=20)
    for s in rs:
        plt.text(tnpos.T[0][s].T, tnpos.T[1][s].T,s+1,color='black',size=22)
    for t in rt:
        plt.text(tnpos.T[0][t].T, tnpos.T[1][t].T,t+1,color='black',size=22)
    plt.savefig(figname+'_domain_net.png',dpi=300,transparent=True)
    plt.show()
    resnet1=maxf_resnet_pd[maxf_resnet_pd>0].fillna(-1)
    resnet2=resnet1[resnet1==-1].fillna(1)
    resnet2[resnet2!=-1].sum().sum()
    print(len(lines_r),resnet2[resnet2!=-1].sum().sum(),'MaxF_sum:',pd.DataFrame(maxf_netsvalue,index=['MaxF']).sum().sum(),'MaxF:',maxf_netsv)
    print('Color Code:',dom_colors)
    pd.DataFrame(maxf_netsvalue,index=['MaxF']).T.sort_values('MaxF',ascending=False).plot.bar(figsize=(14,3),ylim=(0,220))
    return lines_r,lines_g,maxf_nets,maxf_netsvalue,tnpos