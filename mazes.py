from pynput.keyboard import Listener
import numpy as np

class Maze:
    def __init__(self,n,m) -> None:
        self.n = n
        self.m = m
        self.weights = np.random.random((n,m,2)) # 0 : 'down', 1 : 'right'
        self.weights[-1,:,0] = 1 # 1 for wall, 2 for passage
        self.weights[:,-1,1] = 1 # 1 for wall, 2 for passage
        self.V = set([divmod(i,m) for i in range(n*m)])
        self.grow_tree()
        self.string = self.generate_str()
    
    def __str__(self):
        return self.string

    def grow_tree(self):
        x,y,z = np.unravel_index(np.argmin(self.weights), self.weights.shape)
        x,y,z = x.item(), y.item(), z.item()
        # added_edges = {(x,y,z)}
        self.grow_leaf(x,y,z)
        neighbors = self.find_neighbors(x,y,set())
        neighbors = self.find_neighbors(x-(z-1),y+z,neighbors)

        while len(self.V) > 0:
            list_n = list(neighbors)
            # weights = self.weights[*zip(*list_n)]
            x,y,z = list_n[np.argmin(self.weights[*zip(*list_n)])]
            self.grow_leaf(x,y,z)
            neighbors.discard((x,y,z))
            neighbors = self.find_neighbors(x,y,neighbors)
            neighbors = self.find_neighbors(x-(z-1),y+z,neighbors)
        
        self.weights = np.where(self.weights<1,1,self.weights)-1 # re-code with 0 : wall, 1 : passage

    
    def grow_leaf(self,x,y,z):
        self.V.difference_update({(x,y),(x-(z-1),y+z)})
        self.weights[x,y,z] = 2 # 1 for wall, 2 for passage

    def find_neighbors(self,x,y,N):
        neighbors = set()
        if x>0 and (x-1,y) in self.V:
            neighbors.add((x-1,y,0))
        if y>0 and (x,y-1) in self.V:
            neighbors.add((x,y-1,1))
        if (x+1,y) in self.V:
            neighbors.add((x,y,0))
        if (x,y+1) in self.V:
            neighbors.add((x,y,1))
        return N.difference({(x-1,y,0),(x,y-1,1),(x,y,0),(x,y,1)}).union(neighbors)
    
    def generate_str(self):
        string = ' _'*self.m + ' \n|'
        hh = np.where(self.weights[:,:,0]==0,'_',' ')
        vv = np.where(self.weights[:,:,1]==0,'|',' ')
        mat = np.strings.add(hh,vv)
        sep = np.full((self.n,1),'\n|')
        string += ''.join(np.concatenate((mat,sep),1).flatten())
        return string[:-1]

    def Kruskal_partial(self):
        minv = np.min(self.vert)
        minh = np.min(self.horiz)
        first = True

        while len(self.V) > 0:
            if minv<minh:
                pos = np.where(self.vert==minv)
                x,y = pos[0][0],pos[1][0]
                v1,v2 = x*self.m+y, (x+1)*self.m+y

                if (x<self.n-1) and (((v1 in self.V) ^ (v2 in self.V)) or first):
                    self.V -= {v1,v2}
                    self.vert[x,y] = 2
                    if first: first = False
                else:
                    self.vert[x,y] = 1
                minv = np.min(self.vert)
            else:
                pos = np.where(self.horiz==minh)
                x,y = pos[0][0],pos[1][0]
                v1,v2 = x*self.m+y, x*self.m+y+1
                if (y<self.m-1) and (((v1 in self.V) ^ (v2 in self.V)) or first):
                    self.V -= {v1,v2}
                    self.horiz[x,y] = 2
                    if first: first = False
                else:
                    self.horiz[x,y] = 1
                minh = np.min(self.horiz)
        
        self.vert = np.where(self.vert<1,1,self.vert)
        self.horiz = np.where(self.horiz<1,1,self.horiz)
        self.vert -= 1
        self.horiz -= 1
    
    def generate_str_old(self):
        string = '._'*self.m + '.\n'
        hh = np.where(self.vert==1,' ','_')
        vv = np.where(self.horiz==1,'.','|')
        mat = np.strings.add(hh,vv)
        rows = np.array(['|']*self.n)
        
        for i in range(self.m): rows = rows + mat[:,i]

        rows = rows + np.array(['\n']*self.n)

        for i in range(len(rows)): string += rows[i]

        return string

def main():
    M = Maze(int(input("Height of the maze: ")),int(input("Width of the maze: ")))
    print(M)

if __name__=='__main__':
    main()