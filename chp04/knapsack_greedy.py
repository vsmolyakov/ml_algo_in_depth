class Item:       
    def __init__(self, wt, val, ind): 
        self.wt = wt 
        self.val = val 
        self.ind = ind 
        self.cost = val // wt 
  
    def __lt__(self, other): 
        return self.cost < other.cost 
  
class FractionalKnapSack:         
    def get_max_value(self, wt, val, capacity): 
          
        item_list = [] 
        for i in range(len(wt)): 
            item_list.append(Item(wt[i], val[i], i)) 
  
        # sorting items by cost heuristic
        item_list.sort(reverse = True)  #O(nlogn)
  
        total_value = 0
        for i in item_list: 
            cur_wt = int(i.wt) 
            cur_val = int(i.val) 
            if capacity - cur_wt >= 0: 
                capacity -= cur_wt 
                total_value += cur_val 
            else: 
                fraction = capacity / cur_wt 
                total_value += cur_val * fraction 
                capacity = int(capacity - (cur_wt * fraction)) 
                break
        return total_value 
  
if __name__ == "__main__": 
    wt = [10, 20, 30] 
    val = [60, 100, 120] 
    capacity = 50
  
    fk = FractionalKnapSack()  
    max_value = fk.get_max_value(wt, val, capacity)
    print("greedy fractional knapsack") 
    print("maximum value: ", max_value) 

