


class rps_player:

    """
    hand is whatever they'll throw

    """
    def __init__(self,hand,score = 0):
        if hand not in ['r','p','s']:
            self._hand = 'r'
        else:
            self._hand = hand
        self._score = score

    def copy(self):
            return rps_player(self._hand, self._score)
            
    def set_hand(self,hand):
        self._hand = hand

    def score(self):
        return self._score


    def __str__(self):
        return self._hand + ' ' + str(self._score)

    def __or__(self, other):
        if self._hand == 'r':
            if other._hand == 'r':
                self._score += 0
                other._score += 0
            elif other._hand == 'p':
                self._score += -1
                other._score += 1
            else:
                self._score += 1
                other._score += -1
        elif self._hand == 'p':
            if other._hand == 'r':
                self._score += 1
                other._score += -1
            elif other._hand == 'p':
                self._score += 0
                other._score += 0
            else:
                self._score += -1
                other._score += 1
        else:
            if other._hand == 'r':
                self._score += -1
                other._score += 1
            elif other._hand == 'p':
                self._score += 1
                other._score += -1
            else:
                self._score += 0
                other._score += 0
    

def main():
    p1 = rps_player('r')
    p2 = rps_player('p', 8)
    print("Player 1:",p1)
    print("Player 2:",p2)
    p1 | p2
    p2 | p1
    print("Player 1:",p1)
    print("Player 2:",p2)
    p2.set_hand('s')
    p1 | p2
    p2 | p1
    print("Player 1:",p1)
    print("Player 2:",p2)
    p2 = p1.copy()
    p1 | p2
    p2 | p1
    print("Player 1:",p1)
    print("Player 2:",p2)
    p2.set_hand('p')
    p1 | p2
    p2 | p1
    print("Player 1:",p1)
    print("Player 2:",p2)
    

if __name__ == '__main__':
    main()
        
    