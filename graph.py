""" A Python Class
A simple Python graph class, demonstrating the essential 
facts and functionalities of graphs.

Compatible networkx VERSION 2
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
from collections import defaultdict


class Graph(object):

    def __init__(self):
        """ initializes a graph object 
            If no dictionary or None is given, 
            an empty dictionary will be used
        """
        self.nx_graph=nx.Graph()
        self.name='A graph as no name'

    def __eq__(self, other) : 
        #print('yo method')
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(str(self))

    def nodes(self):
        """ returns the vertices of a graph """
        return dict(self.nx_graph.nodes())

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
        #edge = set(edge)
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
        self.nx_graph.node[str(node1)+'bilink'+str(node2)].update({'attr_name':(x1+x2)/2})

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



    def add_uniform_to_leaves(self,leaves,a,b):
        d=dict((leaf,np.random.uniform(a,b)) for leaf in leaves)
        self.add_attibutes(d)

    def create_uniform_leaves(self,names,a,b):
        self.add_nodes(names)
        self.add_uniform_to_leaves(names,a,b)

    def create_classes_uniform_leaves(self,nLeaves,classes):
        names=[0]
        classe,a,b=classes      
        names=[classe+str(i+names[::-1][0]) for i in range(nLeaves)] #pour que les noms soient distincts
        self.create_uniform_leaves(names,a,b)

    def find_leaf(self,beginwith): #assez nulle comme recherche
        nodes=self.nodes()
        returnlist=list()
        for nodename in nodes :
            if str(nodename).startswith(beginwith):
                returnlist.append(nodename)
        return returnlist
    

    def smallest_path(self,start_vertex, end_vertex):

        try:
            shtpath=nx.shortest_path(self.nx_graph,start_vertex,end_vertex)
            return shtpath
        except nx.exception.NetworkXNoPath:
            raise NoPathException('No path between two nodes, graph name : ',self.name)

        

    def attribute_distance(self,node1,node2):

        attr1=self.nx_graph.node[node1]
        attr2=self.nx_graph.node[node2]
        if 'attr_name' in attr1 and 'attr_name' in attr2:

            x1=np.array(attr1['attr_name']).reshape(-1,1)
            x2=np.array(attr2['attr_name']).reshape(-1,1)
            d=float(ot.dist(x1,x2)) #flemme
        return d



    def all_attribute_distance(self,nodeOfInterest=None): #un peu beaucoup vener

        if nodeOfInterest==None :
            v=self.nx_graph.nodes()
        else:
            v=list(set(self.nx_graph.nodes()).intersection(set(nodeOfInterest)))
        pairs = list(itertools.combinations(v,2))
        dist_dic=dict()

        for node1,node2 in pairs:
 
            dist_dic[(node1,node2)]=self.attribute_distance(node1,node2)

        self.dist_dic=dist_dic
        self.max_attr_distance=max(list(dist_dic.values()))

    def vertex_distance(self,start_vertex, end_vertex,method='shortest_path'): #faire une classe distance entre les noeuds

        if method=='shortest_path':
            try :
                dist=len(self.smallest_path(start_vertex, end_vertex))-1
            except NoPathException:
                dist=float('inf')
            return dist
        if method=='weighted_shortest_path':
            try :
                sp=len(self.smallest_path(start_vertex, end_vertex))-1
                if (start_vertex,end_vertex) in self.dist_dic:
                    d=self.dist_dic[(start_vertex,end_vertex)]
                elif (end_vertex,start_vertex) in self.dist_dic:
                    d=self.dist_dic[(end_vertex,start_vertex)]
                else :
                    d=np.nan
                maxd=self.max_attr_distance
                dist=sp*d/maxd
            except NoPathException:
                dist = float('inf')
            return dist


    def distance_matrix(self,nodeOfInterest=None,method='shortest_path',changeInf=True,maxvaluemulti=10):
        if nodeOfInterest==None :
            v=self.nx_graph.nodes()
        else:
            v=list(set(self.nx_graph.nodes()).intersection(set(nodeOfInterest)))
        self.map_node=dict([i for i in enumerate(v)]) # à créer ailleurs 
        self.inv_map_node = {v: k for k, v in self.map_node.items()} # à créer ailleurs 
        pairs = list(itertools.combinations(v,2))
        C=np.zeros((len(v),len(v)))
        if method=='weighted_shortest_path':
            self.all_attribute_distance(nodeOfInterest)
        for (s,e) in pairs:
            distance=self.vertex_distance(s,e,method=method)
            C[self.inv_map_node[s]-1,self.inv_map_node[e]-1]=distance
        #print(C)
        C=C+C.T

        if changeInf==True:
            C[C==float('inf')]=maxvaluemulti*np.max(C) # à voir

        return C

    def display_graph(self,**kwargs):
        # Ne marche pas avec les selfs loops
        nx.draw_networkx(self.nx_graph,**kwargs)
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
        neighbors = list(G.neighbors(root))
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
        #return [x for x in T.nodes_iter() if T.out_degree(x)==0 and T.in_degree(x)==1]
        return [x for x in T.nodes if T.out_degree(x)==0 and T.in_degree(x)==1]

    def construct_tree(self):
        self.tree=nx.bfs_tree(self.nx_graph, 1) #create trees

    def leaves_matrix_attr(self):
        leaves=self.return_leaves(self.tree)
        d=dict((k, v) for k, v in self.nx_graph.node.items() if k in set(leaves))
        x=[]
        for k,v in d.items():
            x.append(v['attr_name'])

        return np.array(x)

    def all_matrix_attr(self):
        d=dict((k, v) for k, v in self.nx_graph.node.items())
        x=[]
        for k,v in d.items():
            x.append(v['attr_name'])

        return np.array(x)


class NoPathException(Exception):
    pass


# C'est moche mais ça fait le taf :

def generate_binary_uniform_tree(maxdepth,coupling='cross',a=0,b=5,c=5,d=10):#il faut que nlowLeaves soit une puissance de 2
    graph=Graph()
    #randint=np.random.randint(2,high=maxdepth)
    randint=maxdepth
    nlowLeaves=2**randint
    groupe=('A',a,b)
    graph.create_classes_uniform_leaves(nlowLeaves,groupe)
    groupe=('B',c,d)
    graph.create_classes_uniform_leaves(nlowLeaves,groupe)
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

def build_binary_uniform_dataset(nTree1,nTree2,maxdepth,a=0,b=5,c=5,d=10):
    data=[]
    for i in range(nTree1):
        data.append((generate_binary_uniform_tree(maxdepth,coupling='cross',a=a,b=b,c=c,d=d),0))
    for i in range(nTree2):
        data.append((generate_binary_uniform_tree(maxdepth,coupling='nocross',a=a,b=b,c=c,d=d),1))

    return data

def build_one_tree_dataset_from_xml(path,classe,max_depth):
    onlyfiles = utils.read_files(path)
    data=[]
    for f in onlyfiles :
        G=Graph()
        G.build_Xml_tree(path+'/'+f,max_depth)
        data.append((G,classe)) 

    return data

def build_one_enzyme_graph(path,file):

    with open(path+file) as f:
        sections = list(utils.per_section(f,lambda x:x.startswith('#')))
        graph=Graph()
        k=1
        for label in sections[0]:
            graph.add_vertex(k)
            graph.add_one_attribute(k,label)
            k=k+1
        k=1
        for edges in sections[1]:
            for node in edges.split(','):
                if node!='':
                    graph.add_edge((k,int(node)))
            k=k+1
        classe=int(sections[2][0])
        graph.name=file

    return graph,classe

def build_one_mutag_graph(path,file):

    with open(path+file) as f:
        sections = list(utils.per_section(f,lambda x:x.startswith('#')))
        graph=Graph()
        k=1
        for label in sections[0]:
            graph.add_vertex(k)
            graph.add_one_attribute(k,label)
            k=k+1
        k=1
        for edges in sections[1]:           
            node1=edges.split(',')[0]
            node2=edges.split(',')[1]
            graph.add_edge((int(node1),int(node2)))
            k=k+1
        classe=int(sections[2][0])
        graph.name=file

    return graph,classe

def build_mutag_dataset(path):

    data=[]
    y=[]
    for file in utils.read_files(path):
        graph,classe=build_one_mutag_graph(path,file)
        data.append(graph)
        y.append(classe)

    return zip(data,y)


def build_enzyme_dataset(path):

    data=[]
    y=[]
    for file in utils.read_files(path):
        graph,classe=build_one_enzyme_graph(path,file)
        data.append(graph)
        y.append(classe)

    return list(zip(data,y))

def node_labels_dic(path,name):
    node_dic=dict()
    with open(path+name) as f:
        sections = list(utils.per_section(f))
        k=1
        for elt in sections[0]:
            node_dic[k]=int(elt)
            k=k+1
    return node_dic

def node_attr_dic(path,name):
    node_dic=dict()
    with open(path+name) as f:
        sections = list(utils.per_section(f))
        k=1
        for elt in sections[0]:
            node_dic[k]=[float(x) for x in elt.split(',')]
            k=k+1
    return node_dic

def graph_label_list(path,name):
    graphs=[]
    with open(path+name) as f:
        sections = list(utils.per_section(f))
        k=1
        for elt in sections[0]:
            graphs.append((k,int(elt)))
            k=k+1
    return graphs
def graph_indicator(path,name):
    data_dict = defaultdict(list)
    with open(path+name) as f:
        sections = list(utils.per_section(f))
        k=1
        for elt in sections[0]:
            data_dict[int(elt)].append(k)
            k=k+1
    return data_dict

def compute_adjency(path,name):
    adjency= defaultdict(list)
    with open(path+name) as f:
        sections = list(utils.per_section(f))
        for elt in sections[0]:
            adjency[int(elt.split(',')[0])].append(int(elt.split(',')[1]))
    return adjency


def build_NCI1_dataset(path):
    node_dic=node_labels_dic(path,'NCI1_node_labels.txt')
    graphs=graph_label_list(path,'NCI1_graph_labels.txt')
    adjency=compute_adjency(path,'NCI1_A.txt')
    data_dict=graph_indicator(path,'NCI1_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data


def build_reddit_dataset(path):
    graphs=graph_label_list(path,'REDDIT-MULTI-5K_graph_labels.txt')
    adjency=compute_adjency(path,'REDDIT-MULTI-5K_A.txt')
    data_dict=graph_indicator(path,'REDDIT-MULTI-5K_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data

def build_MUTAG_dataset(path):
    graphs=graph_label_list(path,'MUTAG_graph_labels.txt')
    adjency=compute_adjency(path,'MUTAG_A.txt')
    data_dict=graph_indicator(path,'MUTAG_graph_indicator.txt')
    node_dic=node_labels_dic(path,'MUTAG_node_labels.txt') # ya aussi des nodes attributes ! The fuck ?
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data

def build_ENZYMES_dataset(path):
    graphs=graph_label_list(path,'ENZYMES_graph_labels.txt')
    adjency=compute_adjency(path,'ENZYMES_A.txt')
    data_dict=graph_indicator(path,'ENZYMES_graph_indicator.txt')
    node_dic=node_attr_dic(path,'ENZYMES_node_attributes.txt') # ya aussi des nodes attributes ! The fuck ?
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

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




 
    






