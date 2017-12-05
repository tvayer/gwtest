
def wgw_tree_distance(alpha,epsilon,method='shortest_path',features_metric='sqeuclidean'):

    def dist(graph1,graph2):
        import ot
        import WGW_2 as wgw
        import numpy as np

        leaves1=graph1.return_leaves(graph1.tree)
        leaves2=graph2.return_leaves(graph2.tree)
        C1=graph1.distance_matrix(nodeOfInterest=leaves1,method=method)
        C2=graph2.distance_matrix(nodeOfInterest=leaves2,method=method)
        t1masses = np.ones(len(leaves1))/len(leaves1)
        t2masses = np.ones(len(leaves2))/len(leaves2)
        x1=graph1.leaves_matrix_attr().reshape(-1, 1) # A regarder si c'est pas le contraire !!!!!!
        x2=graph2.leaves_matrix_attr().reshape(-1, 1)

        if features_metric=='dirac':
            f=lambda x,y: x==y
            M=ot.dist(x1,x2,metric=f)
        else:
            M=ot.dist(x1,x2,metric=features_metric) 
        M= M/np.max(M)

        transpwgw,log= wgw.wgw(M,C1,C2,t1masses,t2masses,'square_loss',epsilon,alpha,max_iter=500,verbose=False,log=True)

        return log['GW_dist'][::-1][0]
    
    return dist

def wgw_graph_distance(alpha,epsilon,method='shortest_path',features_metric='sqeuclidean'): # il faut que les features aient la même dimension


    def dist(graph1,graph2):

        import ot
        import WGW_2 as wgw
        import numpy as np

        nodes1=graph1.nodes()
        nodes2=graph2.nodes()
        C1=graph1.distance_matrix(method=method)
        C2=graph2.distance_matrix(method=method)
        t1masses = np.ones(len(nodes1))/len(nodes1)
        t2masses = np.ones(len(nodes2))/len(nodes2)
        x1=graph1.all_matrix_attr().reshape(-1, 1)
        x2=graph2.all_matrix_attr().reshape(-1, 1)

        if features_metric=='dirac':
            f=lambda x,y: x==y
            M=ot.dist(x1,x2,metric=f)
        else:
            M=ot.dist(x1,x2,metric=features_metric) 
        M= M/np.max(M)

        transpwgw,log= wgw.wgw(M,C1,C2,t1masses,t2masses,'square_loss',epsilon,alpha,max_iter=500,verbose=False,log=True)

        return log['GW_dist'][::-1][0]
    
    return dist


def emd_tree_distance(graph1,graph2,features_metric='sqeuclidean'):

    import ot
    import numpy as np

    leaves1=graph1.return_leaves(graph1.tree)
    leaves2=graph2.return_leaves(graph2.tree)
    t1masses = np.ones(len(leaves1))/len(leaves1)
    t2masses = np.ones(len(leaves2))/len(leaves2)
    x1=graph1.leaves_matrix_attr().reshape(-1, 1)
    x2=graph2.leaves_matrix_attr().reshape(-1, 1)

    if features_metric=='dirac':
        f=lambda x,y: x==y
        M=ot.dist(x1,x2,metric=f)
    else:
        M=ot.dist(x1,x2,metric=features_metric) 
    M= M/np.max(M)

    transp = ot.emd(t1masses,t2masses, M)

    return np.sum(transp*M)

def emd_graph_distance(graph1,graph2,features_metric='sqeuclidean'):

    import ot
    import numpy as np

    nodes1=graph1.nodes()
    nodes2=graph2.nodes()
    t1masses = np.ones(len(nodes1))/len(nodes1)
    t2masses = np.ones(len(nodes2))/len(nodes2)
    x1=graph1.all_matrix_attr().reshape(-1, 1)
    x2=graph2.all_matrix_attr().reshape(-1, 1)

    if features_metric=='dirac':
        f=lambda x,y: x==y
        M=ot.dist(x1,x2,metric=f)
    else:
        M=ot.dist(x1,x2,metric=features_metric) 
    M= M/np.max(M)

    transp = ot.emd(t1masses,t2masses, M)

    return np.sum(transp*M)