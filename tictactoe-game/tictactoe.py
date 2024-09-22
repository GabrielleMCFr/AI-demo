"""
Tic Tac Toe Player
"""

import math
import copy
import random

X = "X"
O = "O"
EMPTY = None

# basic custom exception
class InvalidAction(Exception):
    pass


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    countx = 0
    counto = 0

    if board == [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY],[EMPTY, EMPTY, EMPTY]]:
        player = "X"

    # if board isnt empty, just count how many Xs and Os there are.
    else:
        for row in range(3):
            for col in range(3):  
                if board[row][col] == 'X':  
                    countx += 1
                elif board[row][col] == 'O':  
                    counto += 1

        if countx > counto:
            player = "O"
        else:
            player= "X"
    return player


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()

    # iterate through each cell to see if it's empty or not.
    for row in range(3):
        for col in range(3):  
            # add the action to the set if it's empty
            if board[row][col] == EMPTY:
                actions.add((row, col))
               
    return actions
                


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # find out current player turn 
    currentplayer = player(board)

    # deep copy the board
    newboard = copy.deepcopy(board)

    # make move in the new board.
    if newboard[action[0]][action[1]] == EMPTY:
        newboard[action[0]][action[1]] = currentplayer
        return newboard
    else:
        raise InvalidAction('Action is not valid')



def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    boardValues = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    # assign values for each cell.
    for row in range(3):
        for col in range(3):  
        # add the action to the set if it's empty
            if board[row][col] == 'X':  
                boardValues[row][col] = 1
            elif board[row][col] == 'O':  
                boardValues[row][col] = -1
    
    countrow = 0
    # check rows
    for row in range(3):
        for col in range(3):  
        # add the action to the set if it's empty
            countrow += boardValues[row][col]
            if col == 2: 
                if countrow == 3:
                    return 'X'
                elif countrow == -3:
                    return 'O'
                else:
                    countrow = 0
    
    countcol = 0
    # check cols
    for col in range(3):
        for row in range(3):  
        # add the action to the set if it's empty
            countcol += boardValues[row][col]
            if row == 2: 
                if countcol == 3:
                    return 'X'
                elif countcol == -3:
                    return 'O'
                else:
                    countcol = 0

    #check diagonals for a winner
    if (boardValues[0][0] + boardValues[1][1] + boardValues[2][2] == 3) or (boardValues[0][2] + boardValues[1][1] + boardValues[2][0] == 3):
        return 'X'
    elif (boardValues[0][0] + boardValues[1][1] + boardValues[2][2] == -3) or (boardValues[0][2] + boardValues[1][1] + boardValues[2][0] == -3):
        return 'O'
    
    #if there is no winner, returns none
    return None



def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # check the board for empty
    emptyCount = 0
    for row in range(3):
        for col in range(3):  
            if board[row][col] == EMPTY:  
                emptyCount += 1
    
    # if board is full, stop now
    if emptyCount == 0:
        return True
    
    # if board isnt full find out if there is a winner
    else:
        actualWinner = winner(board)
        if actualWinner == 'X' or actualWinner == "O":
            return True

        return False
       

    
def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    potentialwinner = winner(board)

    if potentialwinner == 'X':
        return 1
    elif potentialwinner == 'O':
        return -1
    else:
        return 0



def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # if board is terminal return none
    if terminal(board):
        return None
        
    # else return the optimal move
    else:
        # find who's turn it is (who's the ia is).
        iaplayer = player(board)
        if iaplayer == "X":
            ismax = True
        else:
            ismax = False

        # we need to find what bestMove is
        bestMove = None

        # alpha beta pruning
        # alpha keeps track of the maximum score X player can have, and beta the minimum score O can have. 
        # by comparing the two of them, we can not search useless branches
        #initially, alpha = -math.inf, beta = math.inf. (declared in the for loop)  

        # if ia is X
        if ismax:
            # array to stock the moves and their utility to chose the best one
            movesarr = []
            # iterate through every immediate action to know which one is the best
            for action in actions(board):
                # call minvalue to guess what the opponent next move is gonna be. (here opponent is O, so they aim to minimize.) 
                # (then recursively play games until game over)
                # append the moves and their utility to the array to sort them by their utility later.
                movesarr.append((Minalphabeta(result(board, action), -math.inf, math.inf), action))

            #select best move (of one of them) by organizing them by utility value (descendant to have the biggest first)
            movesarr.sort(key=lambda tuple: tuple[0], reverse=True)
            # select the first one of the array (biggest utility) and return the corresponding move.
            bestMove = movesarr[0][1]
            return bestMove

        # if ia is O
        else:
            # array to stock the moves and their utility to chose the best one
            movesarr = []
            # iterate through every immediate action to know which one is the best
            for action in actions(board):
                # call maxvalue to guess what the opponent next move is gonna be. (here opponent is X, so they aim to maximize.) 
                # (then recursively play games until game over)
                # append the moves and their utility to the array to sort them by their utility later.
                movesarr.append((Maxalphabeta(result(board, action), -math.inf, math.inf), action))

            #select best move (of one of them) by organizing them by utility value (ascendant to have the smallest first)
            movesarr.sort(key=lambda tuple: tuple[0])
            # select the first one of the array (smallest utility) and return the corresponding move.
            bestMove = movesarr[0][1]
            return bestMove



def Maxvalue(board):
    # can only do better than this
    val = -math.inf

    # if board if terminal, means games over, so return utility of the board. (recursive endpoint)
    if terminal(board):
        util = utility(board)
        return util
    
    # recursively find outputs for every actions
    for action in actions(board):
        # ping pong between minvalue and maxvalue, every time the player change, until terminal(board)
        val = max(val, Minvalue(result(board, action)))   

    # return biggest utility of the branch
    return val



def Minvalue(board):
    # can only do better than this
    val = math.inf

    # if games over, return the utility of the end board (recursive endpoint)
    if terminal(board):
        # ping pong between minvalue and maxvalue, every time the player change, until terminal(board)
        util = utility(board)
        return util
    
    # recursively find outputs for every actions possible everytime a new cell is played
    for action in actions(board):
        val = min(val, Maxvalue(result(board, action)))     

     # return smallest utility of the branch
    return val



# ALPHA BETA PRUNING (ameliored minimax, replace Minvalue and Maxvalue functions)
def Minalphabeta(board, alpha, beta):
    # can only do better than this
    val = math.inf

    # if games over, return the utility of the end board (recursive endpoint)
    if terminal(board):
        # ping pong between minvalue and maxvalue, every time the player change, until terminal(board)
        util = utility(board)
        return util
    
    # recursively find outputs for every actions possible everytime a new cell is played
    for action in actions(board):
        val = min(val, Maxalphabeta(result(board, action), alpha, beta))   
        # alpha beta pruning : if there already is a better move, dont bother searching. 
        #  meaning here, if the o player score become less than what the x player is assured of,
        # we doesnt need to keep going, since the x player would optimally choose the better choice for them (biggest). 
        beta = min(beta, val)
        if beta <= alpha:
            return val 

     # return smallest utility of the branch
    return val


def Maxalphabeta(board, alpha, beta):
    # can only do better than this
    val = -math.inf

    # if board if terminal, means games over, so return utility of the board. (recursive endpoint)
    if terminal(board):
        util = utility(board)
        return util
    
    # recursively find outputs for every actions
    for action in actions(board):
        # ping pong between minvalue and maxvalue, every time the player change, until terminal(board)
        val = max(val, Minalphabeta(result(board, action), alpha, beta))   

        # alpha beta pruning : if there already is a better move, dont bother searching. 
        #  meaning here, if the x player score become more than what the O player is assured of,
        # we doesnt need to keep going, since the O player would optimally choose the better choice for them (lowest). 
        alpha = max(alpha, val)
        if beta <= alpha:
            return val

    # return biggest utility of the branch
    return val



# don't mind this, it's a ia who choose a random move, for testing purposes
def randomIA(board):
    # if board is terminal return none
    if terminal(board):
        return None
    # else return any move
    else:
        moves = []
        #choose a random move.
        for action in actions(board):
            moves.append(action)
        
        return random.choice(moves)