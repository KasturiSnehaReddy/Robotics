import heapq
import networkx as nx
from path_planning import euclidean_distance

class SpaceTimeAStar:
    def __init__(self, G):
        self.G = G

    def find_path(self, start, goal, constraints, max_time=100):
        # Constraints: 
        # vertex: ((x, y), t)
        # edge: ((from_x, from_y), (to_x, to_y), t)
        open_set = []
        # (f_score, g_score, (x, y, t), parent)
        heapq.heappush(open_set, (euclidean_distance(start, goal), 0, (start[0], start[1], 0), None))
        
        g_score = { (start[0], start[1], 0): 0 }
        came_from = {}
        
        while open_set:
            _, current_g, current_state, parent = heapq.heappop(open_set)
            x, y, t = current_state
            
            # If we reached the goal, ensure we aren't forced to move by future constraints
            if (x, y) == goal:
                # We reached goal, let's assume we stop here.
                # To be rigorous, we should check if resting here violates a constraint later,
                # but for this MVP, we stop.
                path = []
                curr = current_state
                while curr is not None:
                    path.append((curr[0], curr[1]))
                    curr = came_from.get(curr)
                return path[::-1]
            
            if t >= max_time:
                continue
                
            # Possible moves: 4-way + wait
            neighbors = [(nx, ny) for nx, ny in self.G.neighbors((x, y))]
            neighbors.append((x, y)) # Wait in place
            
            for nx_pos in neighbors:
                nx_state = (nx_pos[0], nx_pos[1], t + 1)
                
                # Check vertex constraints
                if nx_state in constraints.get('vertex', set()):
                    continue
                
                # Check edge constraints (swap conflicts)
                edge_constraint = ((x, y), nx_pos, t + 1)
                if edge_constraint in constraints.get('edge', set()):
                    continue
                
                tentative_g = current_g + 1
                if nx_state not in g_score or tentative_g < g_score[nx_state]:
                    came_from[nx_state] = current_state
                    g_score[nx_state] = tentative_g
                    f_score = tentative_g + euclidean_distance(nx_pos, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, nx_state, current_state))
                    
        return [] # No path found

class CBS:
    def __init__(self, G):
        self.G = G
        self.planner = SpaceTimeAStar(G)

    def find_conflict(self, paths):
        # paths is a dict: agent_id -> list of (x,y)
        max_t = max([len(p) for p in paths.values()])
        
        # Check step by step
        for t in range(max_t):
            positions = {}
            for agent, path in paths.items():
                pos = path[t] if t < len(path) else path[-1] # Rest at goal
                
                # Vertex conflict check
                if pos in positions:
                    other_agent = positions[pos]
                    # Return (agent1, agent2, pos, t)
                    return ('vertex', other_agent, agent, pos, t)
                positions[pos] = agent
                
            # Edge conflict check (swap)
            if t > 0:
                for a1, p1 in paths.items():
                    for a2, p2 in paths.items():
                        if a1 >= a2: continue
                        prev1 = p1[t-1] if t-1 < len(p1) else p1[-1]
                        curr1 = p1[t] if t < len(p1) else p1[-1]
                        prev2 = p2[t-1] if t-1 < len(p2) else p2[-1]
                        curr2 = p2[t] if t < len(p2) else p2[-1]
                        
                        if prev1 == curr2 and curr1 == prev2 and prev1 != curr1:
                            return ('edge', a1, a2, prev1, curr1, t)
        return None

    def solve(self, starts, goals):
        # starts: dict agent_id -> (x,y)
        # goals: dict agent_id -> (x,y)
        root = {
            'constraints': {agent: {'vertex': set(), 'edge': set()} for agent in starts},
            'paths': {},
            'cost': 0
        }
        
        for agent in starts:
            path = self.planner.find_path(starts[agent], goals[agent], root['constraints'][agent])
            if not path:
                print(f"No initial path for {agent}")
                return None
            root['paths'][agent] = path
            root['cost'] += len(path)
            
        open_list = [(root['cost'], id(root), root)]
        
        while open_list:
            cost, _, node = heapq.heappop(open_list)
            
            conflict = self.find_conflict(node['paths'])
            if not conflict:
                print("CBS: Found collision-free paths!")
                return node['paths']
                
            # print(f"CBS: Found conflict {conflict}")
            
            # Branching
            ctype = conflict[0]
            if ctype == 'vertex':
                _, a1, a2, pos, t = conflict
                
                # Child 1: a1 cannot be at pos at time t
                child1 = self.create_child(node, a1, 'vertex', (pos[0], pos[1], t), starts, goals)
                if child1: heapq.heappush(open_list, (child1['cost'], id(child1), child1))
                
                # Child 2: a2 cannot be at pos at time t
                child2 = self.create_child(node, a2, 'vertex', (pos[0], pos[1], t), starts, goals)
                if child2: heapq.heappush(open_list, (child2['cost'], id(child2), child2))
                
            elif ctype == 'edge':
                _, a1, a2, u, v, t = conflict
                # Child 1: a1 cannot move u->v at time t
                child1 = self.create_child(node, a1, 'edge', (u, v, t), starts, goals)
                if child1: heapq.heappush(open_list, (child1['cost'], id(child1), child1))
                
                # Child 2: a2 cannot move v->u at time t
                child2 = self.create_child(node, a2, 'edge', (v, u, t), starts, goals)
                if child2: heapq.heappush(open_list, (child2['cost'], id(child2), child2))
                
        return None

    def create_child(self, node, agent, ctype, constraint, starts, goals):
        import copy
        new_node = copy.deepcopy(node)
        new_node['constraints'][agent][ctype].add(constraint)
        
        new_path = self.planner.find_path(starts[agent], goals[agent], new_node['constraints'][agent])
        if not new_path:
            return None
            
        new_node['paths'][agent] = new_path
        new_node['cost'] = sum(len(p) for p in new_node['paths'].values())
        return new_node

if __name__ == "__main__":
    # Quick test
    from path_planning import build_warehouse_graph
    G, _ = build_warehouse_graph()
    starts = {'robot1': (0, -8), 'robot2': (0, 8)}
    goals = {'robot1': (0, 8), 'robot2': (0, -8)}
    
    cbs = CBS(G)
    paths = cbs.solve(starts, goals)
    if paths:
        for a, p in paths.items():
            print(f"{a} path length: {len(p)}")
