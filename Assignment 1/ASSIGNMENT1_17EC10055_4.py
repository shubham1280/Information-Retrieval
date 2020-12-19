from collections import defaultdict
from collections.abc import Set
import json
import glob
import os
import sys
import numpy as np
import pandas as pd

class TrieNode(Set):
    """A set of strings implemented using a trie."""
    def __init__(self, iterable=()):
        self._children = defaultdict(TrieNode)
        self._end = False
        for element in iterable:
            self.add(element)

    def add(self, element):
        node = self
        for s in element:
            node = node._children[s]
        node._end = True

    def __contains__(self, element):
        node = self
        for k in element:
            if k not in node._children:
                return False
            node = node._children[k]
        return node._end

    def search(self, term):
        """Return the elements of the set matching the search term, which may
        include wildcard *
        """
        results = set() # Set of elements matching search term.
        element = []    # Current element reached in search.
        def _search(m, node, i):
            # Having just matched m, search for term[i:] starting at node.
            element.append(m)
            if i == len(term):
                if node._end:
                    results.add(''.join(element))
            elif term[i] == '*':
                _search('', node, i + 1)
                for k, child in node._children.items():
                    _search(k, child, i)
            elif term[i] in node._children:
                _search(term[i], node._children[term[i]], i + 1)
            element.pop()
        _search('', self, 0)
        return results

    def __iter__(self):
        return iter(self.search('*'))

    def __len__(self):
        return sum(1 for _ in self)
if __name__ == "__main__":
    with open("Inverted_Positional_Index.json") as f:
        invert_id = json.load(f)
    root = TrieNode()
    for key,value in invert_id.items():
        root.add(key)
    with open(sys.argv[1]) as f:
        contents = f.read()
    contents = contents.split("\n")
    with open("RESULTS1_17EC10055"+'.txt', 'w') as f:
        for i in range(0,len(contents)):
            print("///////////////////////////////////////////////////")
            print(contents[i],":")
            ans = list(root.search(contents[i]))
            print(len(ans))
            if len(ans)>0:
                for t in range(0,len(ans)):
                    a = ans[t]
                    f.write(a+":")
                    print(a,":",end=" ")
                    for j in range(0,len(invert_id[a])):
                        if j==(len(invert_id[a])-1):
                            if t==(len(ans)-1):
                                f.write("<"+str(invert_id[a][j][0])+","+str(invert_id[a][j][1])+">")
                            else:
                                f.write("<"+str(invert_id[a][j][0])+","+str(invert_id[a][j][1])+">;")
                            print("<",invert_id[a][j][0],",",invert_id[a][j][1],">;")
                        else:
                            f.write("<"+str(invert_id[a][j][0])+","+str(invert_id[a][j][1])+">,")
                            print("<",invert_id[a][j][0],",",invert_id[a][j][1],">,",end=" ")
                f.write("\n")
            else:
                f.write("No results found\n")