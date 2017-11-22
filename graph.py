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

    def binary_link(self,node1,node2):
        self.add_vertex(str(node1)+'bilink'+str(node2)) 
        edge = set((node1,str(node1)+'bilink'+str(node2)))
        self.add_edge(edge)
        edge = set((node2,str(node1)+'bilink'+str(node2)))
        self.add_edge(edge)

    def recursive_binary_link(self,nodes):
        double_list=nodes
        while len(double_list)>=2:
            oldnodes=self.nodes()
            for double in utils.chunks(double_list, 2):
                self.binary_link(double[0],double[1])
            newnodes=self.nodes()
            double_list=list(set(newnodes).difference(set(oldnodes)))



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
    

    def smallest_path(self,G,start_vertex, end_vertex,method='nx'):
        if method=='nx':
            smallest=nx.shortest_path(G,start_vertex,end_vertex)
        else : #ne marche pas à revoir
            paths = self.find_all_paths(start_vertex,end_vertex)
            smallest = sorted(paths, key=len)[0]

        return smallest

    def vertex_distance(self,G,start_vertex, end_vertex,method='nx'):

        return len(self.smallest_path(G,start_vertex, end_vertex,method))-1

    def distance_matrix(self,G,nodeOfInterest=None):
        if nodeOfInterest==None :
            v=G.nodes()
        else:
            v=list(set(G.nodes()).intersection(set(nodeOfInterest)))
        self.map_node=dict([i for i in enumerate(v)]) # à créer ailleurs 
        self.inv_map_node = {v: k for k, v in self.map_node.items()} # à créer ailleurs 
        pairs = list(itertools.combinations(v,2))
        C=np.zeros((len(v),len(v)))
        for (s,e) in pairs:
            distance=self.vertex_distance(G,s,e)
            C[self.inv_map_node[s]-1,self.inv_map_node[e]-1]=distance
        #print(C)
        C=C+C.T

        return C

    def display_graph(self,G,**kwargs):
        # Ne marche pas avec les selfs loops
        nx.draw_networkx(G,**kwargs)
        plt.show()

    def rename_nx_graph(self,G,renamedict):
        return nx.relabel_nodes(G,renamedict)

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

    def basic_rename(self,rootiterable):
        l=[{b:a} for a,b,c in utils.depth_iter(rootiterable)]
        renamedict=dict(ChainMap(*l))
        self.nx_graph=self.rename_nx_graph(self.nx_graph,renamedict)


    def build_Xml_tree(self,file_name,max_depth=5):
        xml_tree = xmlTree.parse(file_name)
        xml_root = xml_tree.getroot()

        self.create_graph_dict_xml(xml_root=xml_root,depth=0,max_depth=max_depth)
        #self.add_nodes_and_edges_from_graph_dict()
        self.basic_rename(xml_root)
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

def generate_binary_gaussian_tree(highpowerof2):#il faut que nlowLeaves soit une puissance de 2
    graph=Graph()
    randint=np.random.randint(2,high=highpowerof2)
    nlowLeaves=2**randint
    groupe=('A',0,5)
    graph.create_classes_gaussian_leaves(nlowLeaves,groupe)
    groupe=('B',5,10)
    graph.create_classes_gaussian_leaves(nlowLeaves,groupe)
    noeudB=graph.find_leaf('B')
    noeudA=graph.find_leaf('A')            
    k=0
    for noeud in noeudA:
        graph.binary_link(noeud,noeudB[k])
        k=k+1
    otherNode=list(set(graph.nodes()).difference(set(noeudB).union(set(noeudA))))
    graph.recursive_binary_link(otherNode)

    return graph




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






