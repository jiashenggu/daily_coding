def constructSExpression(s: str):
    graph = [ [False]*26 for _ in range(26)]
    nodeSet = set()
    E2 = False
    numOfEdges = 0
    for i in range(1,len(s), 6):
        x = ord(s[i])-ord('A')
        y = ord(s[i+2]) - ord('A')
        if graph[x][y]:
            E2 = True
        graph[x][y] = True
        nodeSet.add(s[i])
        nodeSet.add(s[i+2])
        numOfEdges+=1

    E1 = False

    for i in range(26):
        count = 0
        for j in range(26):
            if graph[i][j]:
                count+=1
        if count>2:
            return "E1"

    numRoot = 0
    root = " "
    def detectCycle(c, graph, visited):
        if visited[c - 'A']:
            return True
        visited[c -'A'] = True
        for i in range(26):
            if graph[c-'A'][i]:
                if detectCycle('A'+i, graph, visited):
                    return True
        return False
    for c in nodeSet:
        for i in range(26):
            if graph[i][c-'A']:
                break
            if i==25:
                numRoot+=1
                root = c
                visied = [False]*26
                if detectCycle(c, graph, visied):
                    return 'E3'
    if numRoot==0:
        return "E3"
    if numRoot>1:
        return "E4"

    def getSexpression(root, graph):
        if root==' ':
            return ""
        return 
    return getSexprssion(root, graph)