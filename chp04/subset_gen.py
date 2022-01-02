def search(k, n):
    if (k == n):
        #process subset
        print(subset) 
    else:
        search(k+1, n)
        subset.append(k)
        search(k+1, n)
        subset.pop()
    #end if

def bitseq(n):
    for b in range(1 << n):
        subset = []
        for i in range(n):
            if (b & 1 << i):
                subset.append(i)
        #end for
        print(subset)
    #end for 

if __name__ == "__main__":
    n = 4
    subset = []
    search(0, n) #recursive

    #subset = []
    #bitseq(n)    #iterative
