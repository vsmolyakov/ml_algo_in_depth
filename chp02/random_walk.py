import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)

def rand_walk(num_step, num_iter, moves):
    
    #random walk stats
    square_dist = np.zeros(num_iter)
    weights = np.zeros(num_iter)
        
    for it in range(num_iter):
        
        trial = 0
        i = 1
        
        #iterate until we have a non-crossing random walk
        while i != num_step-1:
            
            #init
            X, Y = 0, 0
            weight = 1
            lattice = np.zeros((2*num_step+1, 2*num_step+1))
            lattice[num_step+1,num_step+1] = 1
            path = np.array([0, 0])
            xx = num_step + 1 + X
            yy = num_step + 1 + Y
            
            print("iter: %d, trial %d" %(it, trial))
            
            for i in range(num_step):
                
                up    = lattice[xx,yy+1]
                down  = lattice[xx,yy-1]
                left  = lattice[xx-1,yy]
                right = lattice[xx+1,yy]
                
                #compute available directions
                neighbors = np.array([1, 1, 1, 1]) - np.array([up, down, left, right])
                
                #avoid self-loops
                if (np.sum(neighbors) == 0):
                    i = 1
                    break
                #end if
                
                #compute importance weights: d0 x d1 x ... x d_{n-1}
                weight = weight * np.sum(neighbors)
                
                #sample a move direction
                direction = np.where(np.random.rand() < np.cumsum(neighbors/float(sum(neighbors))))
                
                X = X + moves[direction[0][0],0]
                Y = Y + moves[direction[0][0],1]
                
                #store sampled path
                path_new = np.array([X,Y])
                path = np.vstack((path,path_new))
                
                #update grid coordinates
                xx = num_step + 1 + X
                yy = num_step + 1 + Y                  
                lattice[xx,yy] = 1                                                                                                                                            
            #end for
            
            trial = trial + 1
        #end while                                
        
        #compute square extension
        square_dist[it] = X**2 + Y**2
        
        #store importance weights
        weights[it] = weight                                                        
    #end for
     
    #compute mean square extension
    mean_square_dist = np.mean(weights * square_dist)/np.mean(weights)
    print("mean square dist: ", mean_square_dist)
    
    #generate plots
    plt.figure()
    for i in range(num_step-1):
        plt.plot(path[i,0], path[i,1], path[i+1,0], path[i+1,1], 'ob')            
    plt.title('random walk with no overlaps')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    plt.figure()
    sns.distplot(square_dist)
    plt.xlim(0,np.max(square_dist))
    plt.title('square distance of the random walk')
    plt.xlabel('square distance (X^2 + Y^2)')
    plt.show()
        
    
if __name__ == "__main__":

    num_step = 150 #number of steps in a random walk
    num_iter = 100 #number of iterations for averaging results
    moves = np.array([[0, 1],[0, -1],[-1, 0],[1, 0]]) #2-D moves

    rand_walk(num_step, num_iter, moves)