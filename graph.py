""" A Python Class
A simple Python graph class, demonstrating the essential 
facts and functionalities of graphs.
"""
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import numpy as np
import xml.etree.ElementTree as xmlTree
from collections import ChainMap
import utils 
import random
import ot
import WGW_2 as wgw

class Graph(object):

    def __init__(self, graph_dict=None):
        """ initializes a graph object 
            If no dictionary or None is given, 
            an empty dictionary will be used
        """
        if graph_dict == None:
            graph_dict = {}
        self.graph_dict = graph_dict # à virer useless le nx_graph plus efficace
        self.nx_graph=nx.Graph()

    def nodes(self):
        """ returns the vertices of a graph """
        return self.nx_graph.nodes()

    def edges(self):
        """ returns the edges of a graph """
        return self.nx_graph.edges()

    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in 
            self.graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary. 
            Otherwise nothing has to be done. 
        """
        if vertex not in self.nodes():
            self.nx_graph.add_node(vertex)

    def add_nodes(self, nodes):
        self.nx_graph.add_nodes_from(nodes)

    def add_edge(self, edge):
        """ assumes that edge is of type set, tuple or list; 
            between two vertices can be multiple edges! 
        """
        edge = set(edge)
        (vertex1, vertex2) = tuple(edge)
        self.nx_graph.add_edge(vertex1,vertex2)

    def add_one_attribute(self,node,attr,attr_name='feature'):
        self.nx_graph.add_node(node,attr_name=attr)

    def add_attibutes(self,attributes):
        attributes=dict(attributes)
        for node,attr in attributes.items():
            self.add_one_attribute(node,attr)

    def get_attr(self,vertex):
        return self.nx_graph.node[vertex]

    def binary_link(self,node1,node2):
        self.add_vertex(str(node1)+'bilink'+str(node2)) 
        edge = set((node1,str(node1)+'bilink'+str(node2)))
        self.add_edge(edge)
        edge = set((node2,str(node1)+'bilink'+str(node2)))
        self.add_edge(edge)
        # ajouter l'attribut moyen ?
        x1=np.array(self.get_attr(node1)['attr_name'])
        x2=np.array(self.get_attr(node2)['attr_name'])
        self.nx_graph.node[str(node1)+'bilink'+str(node2)]={'attr_name':(x1+x2)/2}

    def iterative_binary_link(self,nodes,maxIter=None):
        double_list=nodes
        k=0
        go=True
        while go is True:
            oldnodes=self.nodes()
            for double in utils.chunks(double_list, 2):
                self.binary_link(double[0],double[1])
            newnodes=self.nodes()
            double_list=list(set(newnodes).difference(set(oldnodes)))
            k=k+1
            if maxIter is not None:
                go=len(double_list)>=2 and k<maxIter
            else :
                go=len(double_list)>=2



    def add_gaussian_to_leaves(self,leaves,a,b):
        d=dict((leaf,np.random.uniform(a,b)) for leaf in leaves)
        self.add_attibutes(d)

    def create_gaussian_leaves(self,names,a,b):
        self.add_nodes(names)
        self.add_gaussian_to_leaves(names,a,b)

    def create_classes_gaussian_leaves(self,nLeaves,classes):
        names=[0]
        classe,a,b=classes      
        names=[classe+str(i+names[::-1][0]) for i in range(nLeaves)] #pour que les noms soient distincts
        self.create_gaussian_leaves(names,a,b)

    def find_leaf(self,beginwith): #assez nulle comme recherche
        nodes=self.nodes()
        returnlist=list()
        for nodename in nodes :
            if str(nodename).startswith(beginwith):
                returnlist.append(nodename)
        return returnlist
    

    def smallest_path(self,G,start_vertex, end_vertex):

        return nx.shortest_path(G,start_vertex,end_vertex)

    def attribute_distance(self,G,node1,node2):

        attr1=G.node[node1]
        attr2=G.node[node2]
        if 'attr_name' in attr1 and 'attr_name' in attr2:

            x1=np.array(attr1['attr_name']).reshape(-1,1)
            x2=np.array(attr2['attr_name']).reshape(-1,1)
            d=float(ot.dist(x1,x2)) #flemme
        return d



    def all_attribute_distance(self,G,nodeOfInterest=None): #un peu beaucoup vener

        if nodeOfInterest==None :
            v=G.nodes()
        else:
            v=list(set(G.nodes()).intersection(set(nodeOfInterest)))
        pairs = list(itertools.combinations(v,2))
        dist_dic=dict()

        for node1,node2 in pairs:
 
            dist_dic[(node1,node2)]=self.attribute_distance(G,node1,node2)

        self.dist_dic=dist_dic
        self.max_attr_distance=max(list(dist_dic.values()))

    def vertex_distance(self,G,start_vertex, end_vertex,method='shortest_path'): #faire une classe distance entre les noeuds

        if method=='shortest_path':
            return len(self.smallest_path(G,start_vertex, end_vertex))-1
        if method=='weighted_shortest_path':
            sp=len(self.smallest_path(G,start_vertex, end_vertex))-1
            if (start_vertex,end_vertex) in self.dist_dic:
                d=self.dist_dic[(start_vertex,end_vertex)]
            elif (end_vertex,start_vertex) in self.dist_dic:
                d=self.dist_dic[(end_vertex,start_vertex)]
            else :
                d=np.nan
            maxd=self.max_attr_distance
            return sp*d/maxd


    def distance_matrix(self,G,nodeOfInterest=None,method='shortest_path'):
        if nodeOfInterest==None :
            v=G.nodes()
        else:
            v=list(set(G.nodes()).intersection(set(nodeOfInterest)))
        self.map_node=dict([i for i in enumerate(v)]) # à créer ailleurs 
        self.inv_map_node = {v: k for k, v in self.map_node.items()} # à créer ailleurs 
        pairs = list(itertools.combinations(v,2))
        C=np.zeros((len(v),len(v)))
        for (s,e) in pairs:
            if method=='weighted_shortest_path':
                self.all_attribute_distance(G,nodeOfInterest)
            distance=self.vertex_distance(G,s,e,method=method)
            C[self.inv_map_node[s]-1,self.inv_map_node[e]-1]=distance
        #print(C)
        C=C+C.T

        return C

    def display_graph(self,G,**kwargs):
        # Ne marche pas avec les selfs loops
        nx.draw_networkx(G,**kwargs)
        plt.show()


    def create_graph_dict_xml(self,xml_root,depth,max_depth): #mettre dans la classe pour enlever le graph.
        """Create recursively a tree with via xml file """
        
        if (depth ==0) : #at root
            self.add_vertex(xml_root) 
            self.add_one_attribute(xml_root,utils.extract_attribute(xml_root))       
        
        if depth < max_depth:
            
            for xml_child in xml_root:                
                self.add_vertex(xml_child)
                self.add_one_attribute(xml_child,utils.extract_attribute(xml_child))                
                self.add_edge((xml_child,xml_root))                
                self.create_graph_dict_xml(xml_child,depth+1,max_depth)

    def hierarchy_pos(self,G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, 
                  pos = None, parent = None,rotate=False):
        '''If there is a cycle that is reachable from root, then this will see infinite recursion.
           G: the graph
           root: the root node of current branch
           width: horizontal space allocated for this branch - avoids overlap with other branches
           vert_gap: gap between levels of hierarchy
           vert_loc: vertical location of root
           xcenter: horizontal location of root
           pos: a dict saying where all nodes go if they have been assigned
           parent: parent of this branch.'''
        if pos == None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        neighbors = G.neighbors(root)
        if parent != None:   #this should be removed for directed graphs.
            neighbors.remove(parent)  #if directed, then parent not in neighbors.
        if len(neighbors)!=0:
            dx = width/len(neighbors) 
            nextx = xcenter - width/2 - dx/2
            for neighbor in neighbors:
                nextx += dx
                pos = self.hierarchy_pos(G,neighbor, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx, pos=pos, 
                                    parent = root)
        if rotate==True:
            rotate_pos=dict((k, (-1*v[::-1][0],v[::-1][1])) for k, v in pos.items())   
            return rotate_pos
        else:
            return pos


    def rename_nx_graph(self,G,renamedict):
        return nx.relabel_nodes(G,renamedict)

    def iter_rename(self,rootiterable):
        l=[{b:a} for a,b,c in utils.depth_iter(rootiterable)]
        renamedict=dict(ChainMap(*l))
        self.nx_graph=self.rename_nx_graph(self.nx_graph,renamedict)


    def build_Xml_tree(self,file_name,max_depth=5):
        xml_tree = xmlTree.parse(file_name)
        xml_root = xml_tree.getroot()

        self.create_graph_dict_xml(xml_root=xml_root,depth=0,max_depth=max_depth)
        #self.add_nodes_and_edges_from_graph_dict()
        self.iter_rename(xml_root)
        #print('warn : nx_graph was renamed')
        self.construct_tree()


    def return_leaves(self,T):
        return [x for x in T.nodes_iter() if T.out_degree(x)==0 and T.in_degree(x)==1]

    def construct_tree(self):
        self.tree=nx.bfs_tree(self.nx_graph, 1) #create trees

    def leaves_matrix_attr(self):
        leaves=self.return_leaves(self.tree)
        d=dict((k, v) for k, v in self.nx_graph.node.items() if k in set(leaves))
        x=[]
        for k,v in d.items():
            x.append(v['attr_name'])

        return np.array(x)

# C'est moche mais ça fait le taf :

def generate_binary_gaussian_tree(maxdepth,coupling='cross',a=0,b=5,c=5,d=10):#il faut que nlowLeaves soit une puissance de 2
    graph=Graph()
    #randint=np.random.randint(2,high=maxdepth)
    randint=maxdepth
    nlowLeaves=2**randint
    groupe=('A',a,b)
    graph.create_classes_gaussian_leaves(nlowLeaves,groupe)
    groupe=('B',c,d)
    graph.create_classes_gaussian_leaves(nlowLeaves,groupe)
    noeud_0=graph.find_leaf('A')
    noeud_1=graph.find_leaf('B')
    if coupling=='cross':            
        k=0
        for noeud in noeud_0:
            graph.binary_link(noeud,noeud_1[k])
            k=k+1
    else :
        graph.iterative_binary_link(noeud_0,maxIter=1)
        graph.iterative_binary_link(noeud_1,maxIter=1)
    otherNode=list(set(graph.nodes()).difference(set(noeud_1).union(set(noeud_0))))

    graph.iterative_binary_link(otherNode)
    graph.nx_graph=graph.rename_nx_graph(graph.nx_graph,{max(graph.nodes(), key=len):1}) #renomer la racine
    graph.construct_tree()

    return graph

def build_binary_gaussian_dataset(nTree1,nTree2,maxdepth,a=0,b=5,c=5,d=10):
    data=[]
    for i in range(nTree1):
        data.append((generate_binary_gaussian_tree(maxdepth,coupling='cross',a=a,b=b,c=c,d=d),0))
    for i in range(nTree2):
        data.append((generate_binary_gaussian_tree(maxdepth,coupling='nocross',a=a,b=b,c=c,d=d),1))

    return data

def build_one_tree_dataset_from_xml(path,classe,max_depth):
    onlyfiles = utils.read_files(path)
    data=[]
    for f in onlyfiles :
        G=Graph()
        G.build_Xml_tree(path+'/'+f,max_depth)
        data.append((G,classe)) 

    return data

def split_train_test(dataset,ratio):
    x_train=random.sample(dataset,int(ratio*len(dataset)))
    x_test=list(set(dataset).difference(set(x_train)))

    return x_train,x_test

def build_train_test(list_tree_dataset,ratios): #beurk

    'For a list of DATASET'
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    if type(ratios)==float:
        for dataset in list_tree_dataset:
            train,test=split_train_test(dataset,ratios)
            a,b=zip(*train)
            x_train.append(list(a))
            y_train.append(list(b))
            a,b=zip(*test)
            x_test.append(list(a))
            y_test.append(list(b))


    else :
        k=0
        for dataset in list_tree_dataset:
            train,test=split_train_test(dataset,ratios[k])
            k=k+1
            a,b=zip(*train)
            x_train.append(list(a))
            y_train.append(list(b))
            a,b=zip(*test)
            x_test.append(list(a))
            y_test.append(list(b))

    return sum(x_train,[]),sum(x_test,[]),sum(y_train,[]),sum(y_test,[]) #ELEGANT


def wgw_tree_distance(alpha,epsilon,method='shortest_path'):

    def dist(graph1,graph2):
        leaves1=graph1.return_leaves(graph1.tree)
        leaves2=graph2.return_leaves(graph2.tree)
        C1=graph1.distance_matrix(graph1.nx_graph,nodeOfInterest=leaves1,method=method)
        C2=graph2.distance_matrix(graph2.nx_graph,nodeOfInterest=leaves2,method=method)
        t1masses = np.ones(len(leaves1))/len(leaves1)
        t2masses = np.ones(len(leaves2))/len(leaves2)
        x1=graph1.leaves_matrix_attr().reshape(1, -1)
        x2=graph2.leaves_matrix_attr().reshape(1, -1)

        M=ot.dist(x1,x2)
        M= M/np.max(M)

        transpwgw,log= wgw.wgw(M,C1,C2,t1masses,t2masses,'square_loss',epsilon,alpha,max_iter=500,verbose=False,log=True)

        return log['GW_dist'][::-1][0]
    
    return dist


def emd_tree_distance(graph1,graph2):

    leaves1=graph1.return_leaves(graph1.tree)
    leaves2=graph2.return_leaves(graph2.tree)
    t1masses = np.ones(len(leaves1))/len(leaves1)
    t2masses = np.ones(len(leaves2))/len(leaves2)
    x1=graph1.leaves_matrix_attr().reshape(1, -1)
    x2=graph2.leaves_matrix_attr().reshape(1, -1)

    M=ot.dist(x1,x2)
    M= M/np.max(M)

    transp = ot.emd(t1masses,t2masses, M)

    return np.sum(transp*M)

 
    






