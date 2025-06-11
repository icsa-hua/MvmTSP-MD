import pulp as pl

# Data
V = [1, 2, 4, 5, 6]                   # target nodes
DEPOT = 3
NODES = V + [DEPOT]
K = [0, 1]                   # two UAVs
H = 20                       # horizon (minutes)
T = list(range(H + 1))

# Duration matrix (minutes) – symmetric
D = {i: {j: 0 for j in NODES} for i in NODES}
for i in NODES:
    for j in NODES:
        if i != j:
            D[i][j] = 2 if (i in V and j in V) else 3  # depot legs take 3, target–target 2

# Energy equals duration
E = {i: {j: D[i][j] for j in NODES} for i in NODES}

BAT_MAX = {0: 100, 1: 100}
T_MAX = 20
WAIT = 1

# ---------- decision variables ----------
x = pl.LpVariable.dicts("x",
                        ((i, j, k) for i in NODES for j in NODES for k in K),
                        cat='Binary')

y_dep = pl.LpVariable.dicts("yDep",
                            ((i, j, k, t) for i in NODES for j in NODES for k in K for t in T
                             if i != j and t + D[i][j] <= H),
                            cat='Binary')

busy = pl.LpVariable.dicts("busy", ((k, t) for k in K for t in T), 0, 1, cat='Binary')
wait = pl.LpVariable.dicts("wait", ((k, t) for k in K for t in T), 0, 1, cat='Binary')
atNode = pl.LpVariable.dicts("pres", ((k, i, t) for k in K for i in NODES for t in T), 0, 1, cat='Binary')
soc = pl.LpVariable.dicts("SoC", ((k, t) for k in K for t in T))

routeT = pl.LpVariable.dicts("routeTime", K, 0)
T_star = pl.LpVariable("makespan", 0)

PROB = pl.LpProblem("MV_fullCoverage_timeIndexed", pl.LpMinimize)

# Depot initialisation
for k in K:
    PROB += atNode[k, DEPOT, 0] == 1
    for i in V:
        PROB += atNode[k, i, 0] == 0
    PROB += soc[k, 0] == BAT_MAX[k]

# Link y_dep and x
for k in K:
    for i in NODES:
        for j in NODES:
            if i == j:
                continue
            PROB += pl.lpSum(
                y_dep[i, j, k, t] for t in T if (i, j, k, t) in y_dep
            ) == x[i, j, k]

# Temporal state propagation
for k in K:
    for t in T:
        PROB += busy[k, t] + wait[k, t] == 1

        # busy definition
        PROB += busy[k, t] >= pl.lpSum(
            y_dep[i, j, k, tau]
            for i in NODES for j in NODES if i != j
            for tau in T
            if tau <= t < tau + D[i][j] and (i, j, k, tau) in y_dep
        )

        # presence update
        if t > 0:
            # depot presence
            PROB += atNode[k, DEPOT, t] == pl.lpSum(
                y_dep[i, DEPOT, k, tau]
                for i in NODES if i != DEPOT
                for tau in T
                if tau + D[i][DEPOT] == t and (i, DEPOT, k, tau) in y_dep
            )
            for j in V:
                PROB += atNode[k, j, t] == pl.lpSum(
                    y_dep[i, j, k, tau]
                    for i in NODES if i != j
                    for tau in T
                    if tau + D[i][j] == t and (i, j, k, tau) in y_dep
                )

        # wait after arrival
        PROB += wait[k, t] >= pl.lpSum(
            y_dep[i, j, k, tau]
            for i in NODES for j in V if i != j
            for tau in T
            if tau + D[i][j] == t and (i, j, k, tau) in y_dep
        )

# Presence completeness
for k in K:
    for t in T:
        PROB += pl.lpSum(atNode[k, i, t] for i in NODES) == 1

# Coverage
for k in K:
    for j in V:
        PROB += pl.lpSum(atNode[k, j, t] for t in T) >= 1

# Static flow constraints
for k in K:
    PROB += pl.lpSum(x[DEPOT, j, k] for j in V) == 1
    PROB += pl.lpSum(x[i, DEPOT, k] for i in V) == 1
    PROB += x[DEPOT, DEPOT, k] == 0
    for i in V:
        PROB += pl.lpSum(x[i, j, k] for j in NODES if j != i) == 1
        PROB += pl.lpSum(x[j, i, k] for j in NODES if j != i) == 1
    for i in NODES:
        for j in NODES:
            if i < j and i != j:
                PROB += x[i, j, k] + x[j, i, k] <= 1

# MTZ
n = len(V)
for k in K:
    u = pl.LpVariable.dicts(f"u{k}", V, 1, n, cat='Integer')
    for i in V:
        for j in V:
            if i != j:
                PROB += u[i] - u[j] + n * x[i, j, k] <= n - 1

# Energy and time
for k in K:
    for t in range(1, H + 1):
        PROB += soc[k, t] == soc[k, t - 1] - pl.lpSum(
            E[i][j] * y_dep[i, j, k, t - 1]
            for i in NODES for j in NODES if i != j and (i, j, k, t - 1) in y_dep
        )
        PROB += soc[k, t] >= 0

    PROB += routeT[k] == pl.lpSum(busy[k, t] + wait[k, t] for t in T)
    PROB += routeT[k] <= T_MAX
    PROB += routeT[k] <= T_star

# No simultaneous departures same arc same time
for i in NODES:
    for j in NODES:
        if i == j:
            continue
        for tau in T:
            if tau + D[i][j] > H:
                continue
            PROB += pl.lpSum(
                y_dep[i, j, k, tau] for k in K if (i, j, k, tau) in y_dep
            ) <= 1

# Objective
PROB += T_star

# Solve
PROB.solve(pl.GLPK_CMD(msg=False, options=['--mipgap', '0.0','--seed', '42']))

print("Status:", pl.LpStatus[PROB.status])
print("Makespan:", T_star.value())
if pl.LpStatus[PROB.status] != 'Optimal': 
            print("❌ Problem is not optimal, returning None...")
# Extract paths

def extract_paths(y_dep, D, WAIT, K, DEPOT, H):
    """Return a chronologically ordered path for every UAV."""
    paths = {k: [] for k in K}

    for k in K:
        # gather every departure decision for this UAV
        legs = [(t, i, j) for (i, j, kk, t), var in y_dep.items()
                              if kk == k and var.value() == 1]
        # sort by start time
        legs.sort(key=lambda x: x[0])

        cur_time = 0
        for t_depart, src, trg in legs:
            # idle fill until next departure (should be only at depot)
            while cur_time < t_depart:
                paths[k].append((cur_time, DEPOT, DEPOT))
                cur_time += 1

            # busy minutes along arc src→trg
            dur = D[src][trg]
            for _ in range(dur):
                paths[k].append((cur_time, src, trg))
                cur_time += 1
            # WAIT minutes of hover at trg
            for _ in range(WAIT):
                paths[k].append((cur_time, trg, trg))
                cur_time += 1

        # after last leg, idle-fill to horizon (optional)
        while cur_time <= H:
            paths[k].append((cur_time, DEPOT, DEPOT))
            cur_time += 1

    return paths

paths = extract_paths(y_dep, D, WAIT, K, DEPOT, H)
for k in K:
    print(f"UAV {k} path:")
    for step in paths[k][:25]:
        print(step)

