import numpy as np

def get_bonded_pairs(positions, charges, threshold=1.6):
    bonded = []
    n = len(charges)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < threshold:
                bonded.append((i, j))
    return bonded

def build_parent_tree(n_atoms, bonded_pairs, root=0):
    graph = {i: [] for i in range(n_atoms)}
    for i, j in bonded_pairs:
        graph[i].append(j)
        graph[j].append(i)

    parents = [-1] * n_atoms
    visited = [False] * n_atoms

    def dfs(node, parent):
        visited[node] = True
        parents[node] = parent
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs(neighbor, node)

    dfs(root, -1)
    return parents

def get_joint_sides(nuclear_charges, parents):
    hydrogens = [i for i, z in enumerate(nuclear_charges) if z == 1]
    # 简单分配为左边偶数索引，右边奇数索引
    joints_left = hydrogens[::2]
    joints_right = hydrogens[1::2]
    return joints_left, joints_right

def build_skeleton(nuclear_charges, positions, root=0):
    bonded_pairs = get_bonded_pairs(positions, nuclear_charges)
    parents = build_parent_tree(len(nuclear_charges), bonded_pairs, root)
    joints_left, joints_right = get_joint_sides(nuclear_charges, parents)

    skeleton = {
        "parents": parents,
        "joints_left": joints_left,
        "joints_right": joints_right
    }
    return skeleton
data = np.load("/root/rmd17/npz_data/rmd17_aspirin.npz")  # 你的路径
positions = data["coords"]  # shape (N, 3)
positions = positions[0]
nuclear_charges = data["nuclear_charges"]  # shape (N,)

skeleton = build_skeleton(nuclear_charges, positions)
print("parents =", skeleton["parents"])
print("joints_left =", skeleton["joints_left"])
print("joints_right =", skeleton["joints_right"])
