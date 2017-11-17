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
        return list(self.graph_dict.keys())

    def edges(self):
        """ returns the edges of a graph """
        return self.__generate_edges()

    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in 
            self.graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary. 
            Otherwise nothing has to be done. 
        """
        if vertex not in self.graph_dict:
            self.graph_dict[vertex] = []

    def add_edge(self, edge):
        """ assumes that edge is of type set, tuple or list; 
            between two vertices can be multiple edges! 
        """
        edge = set(edge)
        (vertex1, vertex2) = tuple(edge)
        if vertex1 in self.graph_dict:
            self.graph_dict[vertex1].append(vertex2)
        else:
            self.graph_dict[vertex1] = [vertex2]

    def __generate_edges(self):
        """ A static method generating the edges of the 
            graph "graph". Edges are represented as sets 
            with one (a loop back to the vertex) or two 
            vertices 
        """
        edges = []
        for vertex in self.graph_dict:
            for neighbour in self.graph_dict[vertex]:
                if {neighbour, vertex} not in edges:
                    edges.append({vertex, neighbour})
        return edges
    
    def find_path(self, start_vertex, end_vertex, path=None):
        """ find a path from start_vertex to end_vertex 
            in graph """
        if path == None:
            path = []
        graph = self.graph_dict
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return path
        if start_vertex not in graph:
            return None
        for vertex in graph[start_vertex]:
            if vertex not in path:
                extended_path = self.find_path(vertex, 
                                               end_vertex, 
                                               path)
                if extended_path: 
                    return extended_path
        return None
    
    def find_all_paths(self, start_vertex, end_vertex, path=[]):
        """ find all paths from start_vertex to 
            end_vertex in graph """
        graph = self.graph_dict 
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return [path]
        if start_vertex not in graph:
            return []
        paths = []
        for vertex in graph[start_vertex]:
            if vertex not in path:
                extended_paths = self.find_all_paths(vertex, 
                                                     end_vertex, 
                                                     path)
                for p in extended_paths: 
                    paths.append(p)
        return paths
    

    def smallest_path(self,G,start_vertex, end_vertex,method='nx'):
        if method=='nx':
            smallest=nx.shortest_path(G,start_vertex,end_vertex)
        else : #ne marche pas à revoir
            paths = self.find_all_paths(start_vertex,end_vertex)
            smallest = sorted(paths, key=len)[0]

        return smallest

    def vertex_distance(self,G,start_vertex, end_vertex,method='nx'):

        return len(self.smallest_path(G,start_vertex, end_vertex,method))-1

    def add_nodes_and_edges_from_graph_dict(self):
        tupleEdges=[tuple(dic) for dic in self.edges()]
        badIndices=[i for i, v in enumerate(tupleEdges) if len(v) == 1]
        for idx in badIndices:
            tupleEdges[idx]=tupleEdges[idx]+tupleEdges[idx]        
        self.nx_graph.add_edges_from([tuple(dic) for dic in tupleEdges])
        self.nx_graph.add_nodes_from(nodes=self.nodes())


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

    def add_one_attribute(self,node,attr,attr_name='feature'):
        self.nx_graph.add_node(node,attr_name=attr)

    def add_attibutes(self,attributes):
        for node,attr in attributes:
            self.add_one_attribute(node,attr)

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

    def build_Xml_tree(self,file_name,max_depth=5):
        xml_tree = xmlTree.parse(file_name)
        xml_root = xml_tree.getroot()
        l=[{b:a} for a,b,c in utils.depth_iter(xml_root)]
        renamedict=dict(ChainMap(*l))
        self.create_graph_dict_xml(xml_root=xml_root,depth=0,max_depth=max_depth)
        self.add_nodes_and_edges_from_graph_dict()
        self.nx_graph=self.rename_nx_graph(self.nx_graph,renamedict)
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




