# Simulation Engine & Fate Oracle

This document details the mathematical rules of Conway's Game of Life, the topological implications of a toroidal boundary, and the exact cycle-detection algorithm used by the Fate Oracle.

---

## 1. Conway's Game of Life Rules (B3/S23)

Conway's Game of Life is a 2-dimensional 2-state cellular automaton governed by simple local transition rules applied synchronously to every cell.

Let $c(r, c) \in \{0, 1\}$ denote the state of a cell at row $r$ and column $c$ at generation $t$, where $0$ is dead and $1$ is alive. Let $N(r, c)$ represent the count of live cells in its 8-cell Moore neighborhood:

$$N(r, c) = \sum_{\substack{\Delta r \in \{-1, 0, 1\} \\ \Delta c \in \{-1, 0, 1\} \\ (\Delta r, \Delta c) \neq (0, 0)}} c(r + \Delta r, c + \Delta c)$$

The state of the cell at generation $t + 1$ is determined by:

$$c_{t+1}(r, c) = \begin{cases} 
1 & \text{if } N(r, c) = 3 \\
c_t(r, c) & \text{if } N(r, c) = 2 \\
0 & \text{otherwise (underpopulation if } <2\text{, overpopulation if } >3\text{)}
\end{cases}$$

In Boolean algebra on arrays, this is evaluated vectorially:
$$\text{alive}_{t+1} = (N = 3) \lor (\text{alive}_t \land (N = 2))$$

---

## 2. Toroidal Topology (Periodic Boundaries)

In a conventional grid, cells on the boundaries have fewer than 8 neighbors unless special padding or boundary conditions are introduced. In Life Oracle, the board is configured as a **flat 2-torus** ($T^2 = S^1 \times S^1$):

$$(r, c) \equiv (r \pmod N, c \pmod N)$$

### Key Advantages:
1. **Uniform Neighborhood**: Every cell has exactly 8 neighbors. No conditional branch logic or edge artifacts occur during execution.
2. **Conservation of Traveling Patterns**: Gliders and spaceships never hit a wall or "fall off" into the void; they loop indefinitely across the opposite boundary.
3. **Finite State Space Guarantee**: Because the grid has size $N \times N$ and each cell has 2 states, the total number of possible states is strictly bounded by $2^{N^2}$.

---

## 3. Exact Cycle Detection Algorithm

Since the state space is finite ($2^{1024}$ for a $32 \times 32$ board) and the transition rule is completely deterministic, **every trajectory must either reach an all-dead state or enter a recurring periodic cycle**.

Let $S_t$ be the state of the grid at generation $t$. The sequence:

$$S_0 \to S_1 \to S_2 \to \dots \to S_t$$

must eventually encounter two generations $t_1 < t_2$ such that $S_{t_1} = S_{t_2}$.

### Algorithm Steps:

1. Maintain an associative hash map:
   $$\mathcal{H}: \text{bytes}(S) \mapsto t$$
   mapping the exact byte representation of a grid state to the first generation index it was encountered.
2. At each generation $t \in [0, \text{MAX\_GENERATIONS}]$:
   - **Extinction Check**: If $\sum S_t = 0$, return fate `EXTINCT` with population 0.
   - **Cycle Check**: Look up $\text{bytes}(S_t)$ in $\mathcal{H}$.
     - If $\text{bytes}(S_t)$ was already seen at generation $t_{\text{prev}}$:
       $$\text{period} = t - t_{\text{prev}}$$
       $$\text{settle\_time} = t_{\text{prev}}$$
       - If $\text{period} = 1$: Return `STABLE` (still life).
       - If $\text{period} \ge 2$: Return `OSCILLATING`.
     - Otherwise, record $\mathcal{H}[\text{bytes}(S_t)] = t$.
   - **Transition**: Compute $S_{t+1} = \text{step}(S_t)$.
3. If $t = \text{MAX\_GENERATIONS}$ without a recurring cycle or extinction, return `UNSETTLED`.

```text
Generation 0     --> Hash S_0 --> seen[H(S_0)] = 0
Generation 1     --> Hash S_1 --> seen[H(S_1)] = 1
...
Generation t     --> Hash S_t == H(S_k) (k < t)
                     => Period = t - k
                     => Settle Time = k
                     => If Period == 1: STABLE
                        Else: OSCILLATING
```

---

## 4. Fate Class Definitions

| Class | Internal ID | Meaning | Condition |
|---|---|---|---|
| `EXTINCT` | 0 | Total death | Cell count reaches 0 |
| `STABLE` | 1 | Still life | Repeats with $\text{period} = 1$ and live cells $> 0$ |
| `OSCILLATING` | 2 | Periodic oscillator | Repeats with $\text{period} \ge 2$ |
| `UNSETTLED` | 3 | Limit reached | No repeat found within limit (1,000 generations) |

### Note on Gliders and Space Travel:
On a wrapped $32 \times 32$ torus, a standard glider displaces $(+1, +1)$ every 4 generations. Therefore, an isolated glider will traverse the entire board and return to its exact original coordinates in $4 \times 32 = 128$ generations, registering as an `OSCILLATING` pattern with period 128.
