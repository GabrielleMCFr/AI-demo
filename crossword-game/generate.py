import sys

from crossword import *



class CrosswordCreator():

    def __init__(self, crossword):
        """
        Initializes a new CSP crossword generator.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        returns 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        ensure that each variable is node-consistent by removing any values (words) that do not satisfy the variable's
        unary constraints, specifically ensuring that the word length matches the required length for the variable
        """
        wordsToRemove = list()
        for var in self.domains:
            for word in self.domains[var]:
                count = 0
                if (len(word) != var.length): 
                    count += 1
                if count == len(self.domains[var]):
                    wordsToRemove.append(word)
        
        for var in self.domains:
            for w in wordsToRemove:
                if w in self.domains[var]:
                    self.domains[var].remove(w)
                

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False
        
        i, j = self.crossword.overlaps[x, y]

        wordsToRemove = list()
        
        for w1 in self.domains[x]:
            # we start with 0 possibility since we havent tested anything yet
            count = 0
            for w2 in self.domains[y]:
                # if a value if possible, add to the count / first line : verify its not out of range
                if i < len(w1) and j < len(w2):
                    if w1[i] == w2[j]:
                        count += 1
            
            # if no possibility, remove the value from the variable domain
            if count == 0:
                wordsToRemove.append(w1)
                revised = True

        for w in wordsToRemove:
            self.domains[x].remove(w)  

        return revised

    def ac3(self, arcs=None):
        """
        Returns True if arc consistency is enforced and no domains are empty;
        returns False if one or more domains end up empty.
        """
        if arcs == None:
            arcs = list()
            for x in self.crossword.variables:
                for y in self.crossword.neighbors(x):
                    arcs.append((x, y))
        
        while arcs:
            x, y = arcs.pop()
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False
                for z in self.crossword.neighbors(x) - self.domains[y]:
                    arcs.append((z, x))
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        if len(assignment) == len(self.crossword.variables):
            return True
        return False

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        words = list()
        for var in assignment:
            # check if all values are distinct
            if assignment[var] not in words:
                words.append(assignment[var])
            else:
                return False

            # check if lengths are correct
            if var.length != len(assignment[var]):
                return False
            
            # check if there are no conflicts between neighbors
            for n in self.crossword.neighbors(var):
                i, j = self.crossword.overlaps[var, n]
                if n in assignment and assignment[var][i] != assignment[n][j]:
                    return False
        
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        choices = dict()
        for value in self.domains[var]:
            count = 0
            for n in self.crossword.neighbors(var):
                if n in assignment:
                    continue
                
                for v in self.domains[n]:
                     # if theres an overlaps check if the words can fit or if it clashes
                    if self.crosswords.overlaps[var, n]:
                        i, j = self.crosswords.overlaps[var, n]
                        if assignment[var][i] != v[j]:
                            count += 1
                    elif value == v:
                        count += 1
                    elif len(v) != n.length:
                        count += 1
                
                # count if it rules out the neighbor ? 
            # then select the number of variable it rules out and add it to choice
            choices[value]= count
            
        sortedDict = sorted(choices.items(), key = lambda item: item[1])
        return list(sortedDict.keys())

    def select_unassigned_variable(self, assignment):
        """
        returns an unassigned variable that is not included in assignment. It selects the variable with the fewest 
        remaining values in its domain. In case of a tie, it chooses the variable with the highest degree;
        if there's still a tie, any of the tied variables can be returned
        """
        vardict = dict()
        for var in self.crossword.variables:
            if var in assignment:
                continue
            else:
                vardict[var] = len(self.domains[var])
                #return var
        
        sortedByMin = sorted(vardict.items(), key = lambda item: item[1])      
        
        sortedByHeuristic = sorted(sortedByMin, key = lambda item: len(self.crossword.neighbors(item[0])))
        return sortedByHeuristic[0][0]
        

    def backtrack(self, assignment):
        """
        uses Backtracking Search to attempt to complete a crossword puzzle given a partial assignment,
        which maps variables to words. It returns a complete assignment if possible; if no valid assignment can be found, 
        it returns None
        """
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)
        for word in self.domains[var]:
            new_assignment = assignment.copy()
            new_assignment[var] = word
            if self.consistent(new_assignment):
                result = self.backtrack(new_assignment)
                if result is not None:
                    return result
        return None


def main():

    if len(sys.argv) not in [3, 4]:
        sys.exit("Wrong arguments count")

    # parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # print result
    if assignment is None:
        print("No solution found")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
