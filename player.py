from abc import ABC

class Player(ABC):
    def __init__(self,player_indx=0,name=None):
        self.indx = player_indx
        self.name = name or "nameless"
        
    

