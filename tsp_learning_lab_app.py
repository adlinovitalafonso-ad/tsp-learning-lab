import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from itertools import permutations
from math import factorial
from string import ascii_uppercase
from time import perf_counter


# ======================================================
# PAGE SETTINGS
# ======================================================

st.set_page_config(
    page_title="TSP Learning Lab",
    page_icon="🧭",
    layout="wide"
)

st.title("🧭 Travelling Salesperson Problem — Learning Lab")

st.write(
    """
    Explore the Travelling Salesperson Problem (TSP) from formulation to solution.
    Change the input data, compare exact and heuristic methods, and observe how
    computational effort grows as the number of cities increases.
    """
)


# ======================================================
# CONSTANTS
# ======================================================

MAX_BRUTE_FORCE_N = 9

DEFAULT_COORDINATES = [
    (1.0, 7.0),
    (4.0, 9.0),
    (8.0, 7.0),
    (7.0, 2.0),
    (2.0, 1.0),
    (10.0, 4.0),
    (5.0, 5.0),
    (9.0, 10.0),
    (1.0, 11.0),
    (12.0, 8.0),
    (3.0, 13.0),
    (13.0, 2.0),
]


# ======================================================
# HELPER FUNCTIONS
# ======================================================

def euclidean_distance(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def build_distance_matrix_from_coordinates(coord_df):
    coords = coord_df[["X", "Y"]].to_numpy(dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def route_distance(route, distance_matrix):
    return float(
        sum(
            distance_matrix[route[i], route[i + 1]]
            for i in range(len(route) - 1)
        )
    )


def route_to_text(route, cities):
    return " → ".join(cities[i] for i in route)


def is_symmetric(distance_matrix):
    return np.allclose(distance_matrix, distance_matrix.T, atol=1e-9)


def brute_force_tsp(distance_matrix, start=0):
    n = len(distance_matrix)
    others = [i for i in range(n) if i != start]
    symmetric = is_symmetric(distance_matrix)

    rows = []
    best_route = None
    best_distance = float("inf")

    for perm in permutations(others):
        # For symmetric TSP, a tour and its reverse are equivalent.
        if symmetric and len(perm) > 1 and perm[0] > perm[-1]:
            continue

        route = [start] + list(perm) + [start]
        distance = route_distance(route, distance_matrix)

        rows.append((route, distance))

        if distance < best_distance:
            best_distance = distance
            best_route = route

    return best_route, best_distance, rows


def nearest_neighbour(distance_matrix, start=0):
    n = len(distance_matrix)
    route = [start]
    unvisited = set(range(n))
    unvisited.remove(start)
    current = start
    steps = []

    while unvisited:
        candidates = sorted(
            [(city, float(distance_matrix[current, city])) for city in unvisited],
            key=lambda x: (x[1], x[0])
        )

        next_city, chosen_distance = candidates[0]

        steps.append(
            {
                "current": current,
                "candidates": candidates,
                "chosen": next_city,
                "distance": chosen_distance,
            }
        )

        route.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    route.append(start)
    return route, route_distance(route, distance_matrix), steps


def two_opt(route, distance_matrix):
    """
    Best-improvement 2-opt.
    At each iteration, test all possible 2-opt moves and accept
    the best improving move.
    """
    best_route = route.copy()
    best_distance = route_distance(best_route, distance_matrix)
    steps = []
    iteration = 1

    while True:
        candidate_best_route = None
        candidate_best_distance = best_distance
        candidate_best_swap = None

        # Keep first and last position fixed as the start city.
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                candidate = (
                    best_route[:i]
                    + best_route[i:j + 1][::-1]
                    + best_route[j + 1:]
                )

                candidate_distance = route_distance(candidate, distance_matrix)

                if candidate_distance < candidate_best_distance - 1e-12:
                    candidate_best_route = candidate
                    candidate_best_distance = candidate_distance
                    candidate_best_swap = (i, j)

        if candidate_best_route is None:
            break

        old_route = best_route.copy()
        old_distance = best_distance
        i, j = candidate_best_swap

        removed_edges = (
            (old_route[i - 1], old_route[i]),
            (old_route[j], old_route[j + 1]),
        )

        added_edges = (
            (old_route[i - 1], old_route[j]),
            (old_route[i], old_route[j + 1]),
        )

        steps.append(
            {
                "iteration": iteration,
                "old_route": old_route,
                "new_route": candidate_best_route.copy(),
                "old_distance": old_distance,
                "new_distance": candidate_best_distance,
                "improvement": old_distance - candidate_best_distance,
                "removed_edges": removed_edges,
                "added_edges": added_edges,
            }
        )

        best_route = candidate_best_route
        best_distance = candidate_best_distance
        iteration += 1

    return best_route, best_distance, steps


def draw_network(cities, coordinates, route=None, title="Network"):
    fig, ax = plt.subplots(figsize=(7, 5))

    if coordinates is not None:
        # Draw faint complete graph in background.
        for i in range(len(cities)):
            for j in range(i + 1, len(cities)):
                x1, y1 = coordinates[cities[i]]
                x2, y2 = coordinates[cities[j]]
                ax.plot([x1, x2], [y1, y2], alpha=0.12, linewidth=1)

        # Draw selected route.
        if route is not None:
            for k in range(len(route) - 1):
                city1 = cities[route[k]]
                city2 = cities[route[k + 1]]
                x1, y1 = coordinates[city1]
                x2, y2 = coordinates[city2]
                ax.plot([x1, x2], [y1, y2], linewidth=3)

        for city in cities:
            x, y = coordinates[city]
            ax.scatter(x, y, s=110, zorder=3)
            ax.text(x + 0.12, y + 0.12, city, fontsize=11, zorder=4)

        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.set_title(title)
        ax.grid(alpha=0.15)

    return fig


def format_seconds(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:,.1f} μs"
    if seconds < 1:
        return f"{seconds * 1e3:,.2f} ms"
    if seconds < 60:
        return f"{seconds:,.3f} s"
    if seconds < 3600:
        return f"{seconds / 60:,.2f} min"
    if seconds < 86400:
        return f"{seconds / 3600:,.2f} h"
    if seconds < 365 * 86400:
        return f"{seconds / 86400:,.2f} days"
    return f"{seconds / (365 * 86400):,.2f} years"


# ======================================================
# SIDEBAR — INPUTS
# ======================================================

st.sidebar.header("Problem Inputs")

input_mode = st.sidebar.radio(
    "Choose how to enter distances",
    ["Coordinates", "Distance Matrix"],
    help=(
        "Coordinates automatically generate Euclidean distances. "
        "Distance Matrix lets you type the distances directly."
    ),
)

n = st.sidebar.slider(
    "Number of cities",
    min_value=4,
    max_value=12,
    value=5,
)

cities = list(ascii_uppercase[:n])

start_city = st.sidebar.selectbox(
    "Starting city",
    cities,
    index=0,
)

start_index = cities.index(start_city)

st.sidebar.caption(
    f"Brute force is enabled only up to {MAX_BRUTE_FORCE_N} cities "
    "to keep the app responsive."
)


# ======================================================
# EDITABLE DATA INPUT
# ======================================================

if input_mode == "Coordinates":
    st.sidebar.subheader("Editable Coordinates")

    default_coord_df = pd.DataFrame(
        DEFAULT_COORDINATES[:n],
        columns=["X", "Y"],
        index=cities,
    )
    default_coord_df.index.name = "City"

    coord_df = st.sidebar.data_editor(
        default_coord_df,
        key=f"coord_editor_{n}",
        use_container_width=True,
        num_rows="fixed",
    )

    coord_df = coord_df[["X", "Y"]].astype(float)
    distance_matrix = build_distance_matrix_from_coordinates(coord_df)

    coordinates = {
        city: (float(coord_df.loc[city, "X"]), float(coord_df.loc[city, "Y"]))
        for city in cities
    }

else:
    st.sidebar.subheader("Editable Distance Matrix")

    # Use Euclidean defaults so the matrix starts with sensible values.
    base_coord_df = pd.DataFrame(
        DEFAULT_COORDINATES[:n],
        columns=["X", "Y"],
        index=cities,
    )
    default_matrix = build_distance_matrix_from_coordinates(base_coord_df)
    default_matrix_df = pd.DataFrame(
        np.round(default_matrix, 2),
        index=cities,
        columns=cities,
    )

    matrix_df = st.sidebar.data_editor(
        default_matrix_df,
        key=f"matrix_editor_{n}",
        use_container_width=True,
        num_rows="fixed",
    )

    matrix_df = matrix_df.astype(float)

    # Enforce zero diagonal.
    for city in cities:
        matrix_df.loc[city, city] = 0.0

    make_symmetric = st.sidebar.checkbox(
        "Force symmetric distances",
        value=True,
        help="If checked, d(i,j) and d(j,i) are averaged.",
    )

    distance_matrix = matrix_df.to_numpy(dtype=float)

    if make_symmetric:
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0.0)

    coordinates = None


distance_df = pd.DataFrame(
    np.round(distance_matrix, 2),
    index=cities,
    columns=cities,
)


# ======================================================
# COMPUTE HEURISTICS
# ======================================================

nn_start_time = perf_counter()
nn_route, nn_distance, nn_steps = nearest_neighbour(
    distance_matrix,
    start=start_index,
)
nn_time = perf_counter() - nn_start_time

opt_start_time = perf_counter()
opt_route, opt_distance, opt_steps = two_opt(
    nn_route,
    distance_matrix,
)
opt_time = perf_counter() - opt_start_time


# ======================================================
# BRUTE FORCE — ONLY FOR SMALL N
# ======================================================

brute_available = n <= MAX_BRUTE_FORCE_N

bf_route = None
bf_distance = None
bf_rows = []
bf_time = None

if brute_available:
    bf_start_time = perf_counter()
    bf_route, bf_distance, bf_rows = brute_force_tsp(
        distance_matrix,
        start=start_index,
    )
    bf_time = perf_counter() - bf_start_time


# ======================================================
# TABS
# ======================================================

tabs = st.tabs(
    [
        "1. Problem & Data",
        "2. Formulation",
        "3. Brute Force",
        "4. Nearest Neighbour",
        "5. 2-opt",
        "6. Compare Methods",
        "7. Complexity",
        "8. Interpretation",
    ]
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = tabs


# ======================================================
# TAB 1 — PROBLEM & DATA
# ======================================================

with tab1:
    st.header("Problem and Input Data")

    st.write(
        f"""
        A salesperson starts from **City {start_city}**, visits every other city
        exactly once, and returns to **City {start_city}**.

        The objective is to minimize total travel distance.
        """
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if input_mode == "Coordinates":
            st.subheader("Coordinates")
            st.dataframe(coord_df, use_container_width=True)
        else:
            st.subheader("Input Type")
            st.info(
                "You are entering the distance matrix directly. "
                "A geometric network is therefore not drawn."
            )

    with col2:
        st.subheader("Distance Matrix")
        st.dataframe(distance_df, use_container_width=True)

    if coordinates is not None:
        st.subheader("Network")
        fig = draw_network(cities, coordinates, title="Complete Network")
        st.pyplot(fig)
        plt.close(fig)

    if is_symmetric(distance_matrix):
        st.success("This is a symmetric TSP: d(i,j) = d(j,i).")
    else:
        st.warning(
            "This is an asymmetric TSP: travelling from i to j may cost "
            "something different from travelling from j to i."
        )


# ======================================================
# TAB 2 — FORMULATION
# ======================================================

with tab2:
    st.header("Mathematical Formulation")

    st.subheader("Decision variable")
    st.latex(
        r"""
        x_{ij} =
        \begin{cases}
        1, & \text{if the tour travels directly from city } i \text{ to city } j\\
        0, & \text{otherwise}
        \end{cases}
        """
    )

    st.subheader("Objective")
    st.latex(
        r"""
        \min Z =
        \sum_i \sum_{\substack{j\\j\neq i}} c_{ij}x_{ij}
        """
    )
    st.write("Minimize the total travel distance.")

    st.subheader("Leave each city exactly once")
    st.latex(
        r"""
        \sum_{\substack{j\\j\neq i}} x_{ij}=1
        \qquad \forall i
        """
    )

    st.subheader("Enter each city exactly once")
    st.latex(
        r"""
        \sum_{\substack{i\\i\neq j}} x_{ij}=1
        \qquad \forall j
        """
    )

    st.subheader("Binary restriction")
    st.latex(r"x_{ij}\in\{0,1\}")

    st.warning(
        """
        These degree constraints alone can produce multiple smaller cycles
        (subtours). A complete TSP model therefore also requires subtour
        elimination constraints.
        """
    )

    st.subheader("One possible subtour-elimination approach: MTZ")
    st.latex(
        r"""
        u_i-u_j+n x_{ij}\le n-1
        """
    )

    st.write(
        """
        The additional ordering variables $u_i$ help prevent disconnected
        cycles and force the selected edges to form one complete tour.
        """
    )


# ======================================================
# TAB 3 — BRUTE FORCE
# ======================================================

with tab3:
    st.header("Brute-Force Enumeration")

    st.write(
        """
        Brute force checks every feasible ordering of the cities and keeps the
        shortest complete tour. It is exact, but its computational effort grows
        factorially.
        """
    )

    if not brute_available:
        st.warning(
            f"""
            Brute force is disabled for **{n} cities** in this app.

            To keep the classroom app responsive, exact enumeration is limited
            to **{MAX_BRUTE_FORCE_N} cities**. Reduce the number of cities in the
            sidebar to inspect all tours.
            """
        )
    else:
        bf_table = pd.DataFrame(
            {
                "Route": [route_to_text(route, cities) for route, _ in bf_rows],
                "Total Distance": [round(distance, 3) for _, distance in bf_rows],
            }
        ).sort_values("Total Distance").reset_index(drop=True)

        st.metric("Tours examined", f"{len(bf_table):,}")
        st.metric("Computation time", format_seconds(bf_time))

        st.subheader("All tours")
        st.dataframe(bf_table, use_container_width=True)

        st.success(
            f"""
            **Optimal route**

            {route_to_text(bf_route, cities)}

            **Minimum distance = {bf_distance:.3f}**
            """
        )

        if coordinates is not None:
            fig = draw_network(
                cities,
                coordinates,
                route=bf_route,
                title="Brute-Force Optimal Tour",
            )
            st.pyplot(fig)
            plt.close(fig)


# ======================================================
# TAB 4 — NEAREST NEIGHBOUR
# ======================================================

with tab4:
    st.header("Nearest Neighbour Heuristic")

    st.write(
        """
        Nearest Neighbour is a **constructive heuristic**.

        From the current city, choose the nearest unvisited city. Repeat until
        every city has been visited, then return to the starting city.
        """
    )

    st.info(
        f"Starting city: **{start_city}**. Change it in the sidebar and the route will recalculate."
    )

    for step_number, step in enumerate(nn_steps, start=1):
        current_city = cities[step["current"]]
        chosen_city = cities[step["chosen"]]

        with st.expander(
            f"Step {step_number}: {current_city} → {chosen_city}",
            expanded=(step_number == 1),
        ):
            candidate_df = pd.DataFrame(
                [
                    {
                        "Candidate City": cities[city],
                        "Distance from Current City": round(distance, 3),
                    }
                    for city, distance in step["candidates"]
                ]
            )

            st.write(f"Current city: **{current_city}**")
            st.dataframe(candidate_df, hide_index=True, use_container_width=True)

            st.success(
                f"Choose **{chosen_city}**, the nearest unvisited city "
                f"(distance = {step['distance']:.3f})."
            )

    st.subheader("Nearest Neighbour solution")
    st.write(f"**Route:** {route_to_text(nn_route, cities)}")

    col1, col2 = st.columns(2)
    col1.metric("Total distance", f"{nn_distance:.3f}")
    col2.metric("Computation time", format_seconds(nn_time))

    if coordinates is not None:
        fig = draw_network(
            cities,
            coordinates,
            route=nn_route,
            title="Nearest Neighbour Tour",
        )
        st.pyplot(fig)
        plt.close(fig)

    st.warning(
        """
        Nearest Neighbour is fast, but it does **not** guarantee the global
        optimum. A locally attractive choice can make the later part of the
        route more expensive.
        """
    )


# ======================================================
# TAB 5 — 2-OPT
# ======================================================

with tab5:
    st.header("2-opt Local Search")

    st.write(
        """
        2-opt begins with an existing tour — here, the Nearest Neighbour tour.
        It repeatedly removes two edges, reconnects the route in another way,
        and keeps the best improving move.
        """
    )

    st.subheader("Initial solution")
    st.write(f"**Nearest Neighbour route:** {route_to_text(nn_route, cities)}")
    st.metric("Initial distance", f"{nn_distance:.3f}")

    if len(opt_steps) == 0:
        st.info(
            """
            No improving 2-opt move was found.

            The Nearest Neighbour tour is already a **2-opt local optimum**.
            This does not necessarily mean it is the global optimum.
            """
        )
    else:
        st.subheader("Improvement steps")

        for step in opt_steps:
            removed = step["removed_edges"]
            added = step["added_edges"]

            removed_text = (
                f"{cities[removed[0][0]]}–{cities[removed[0][1]]} and "
                f"{cities[removed[1][0]]}–{cities[removed[1][1]]}"
            )

            added_text = (
                f"{cities[added[0][0]]}–{cities[added[0][1]]} and "
                f"{cities[added[1][0]]}–{cities[added[1][1]]}"
            )

            with st.expander(
                f"Iteration {step['iteration']}: improve by {step['improvement']:.3f}",
                expanded=(step["iteration"] == 1),
            ):
                st.write(
                    "**Before:** "
                    + route_to_text(step["old_route"], cities)
                )
                st.write(f"Remove edges: **{removed_text}**")
                st.write(f"Add edges: **{added_text}**")
                st.write(
                    "**After:** "
                    + route_to_text(step["new_route"], cities)
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("Before", f"{step['old_distance']:.3f}")
                c2.metric("After", f"{step['new_distance']:.3f}")
                c3.metric("Improvement", f"{step['improvement']:.3f}")

    st.subheader("Final 2-opt solution")
    st.write(f"**Route:** {route_to_text(opt_route, cities)}")

    distance_saved = nn_distance - opt_distance
    improvement_pct = 0.0 if nn_distance == 0 else 100 * distance_saved / nn_distance

    c1, c2, c3 = st.columns(3)
    c1.metric("Final distance", f"{opt_distance:.3f}")
    c2.metric("Distance saved", f"{distance_saved:.3f}")
    c3.metric("Improvement over NN", f"{improvement_pct:.2f}%")

    st.caption(f"2-opt computation time: {format_seconds(opt_time)}")

    if coordinates is not None:
        fig = draw_network(
            cities,
            coordinates,
            route=opt_route,
            title="Tour After 2-opt",
        )
        st.pyplot(fig)
        plt.close(fig)


# ======================================================
# TAB 6 — COMPARE METHODS
# ======================================================

with tab6:
    st.header("Compare Solution Methods")

    comparison = [
        {
            "Method": "Nearest Neighbour",
            "Distance": nn_distance,
            "Time (seconds)": nn_time,
            "Guaranteed Optimal?": "No",
        },
        {
            "Method": "Nearest Neighbour + 2-opt",
            "Distance": opt_distance,
            "Time (seconds)": nn_time + opt_time,
            "Guaranteed Optimal?": "No",
        },
    ]

    if brute_available:
        comparison.insert(
            0,
            {
                "Method": "Brute Force",
                "Distance": bf_distance,
                "Time (seconds)": bf_time,
                "Guaranteed Optimal?": "Yes",
            },
        )

    comparison_df = pd.DataFrame(comparison)

    if brute_available and bf_distance > 0:
        comparison_df["Gap from Optimum (%)"] = (
            (comparison_df["Distance"] - bf_distance)
            / bf_distance
            * 100
        )
    else:
        comparison_df["Gap from Optimum (%)"] = np.nan

    display_comparison = comparison_df.copy()
    display_comparison["Distance"] = display_comparison["Distance"].round(3)
    display_comparison["Time (seconds)"] = display_comparison["Time (seconds)"].map(
        lambda x: f"{x:.6f}"
    )
    display_comparison["Gap from Optimum (%)"] = display_comparison[
        "Gap from Optimum (%)"
    ].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")

    st.dataframe(
        display_comparison,
        hide_index=True,
        use_container_width=True,
    )

    if brute_available:
        nn_gap = 100 * (nn_distance - bf_distance) / bf_distance
        opt_gap = 100 * (opt_distance - bf_distance) / bf_distance

        st.write(
            f"""
            For this input, Nearest Neighbour is **{nn_gap:.2f}%** above the
            exact optimum, while Nearest Neighbour + 2-opt is **{opt_gap:.2f}%**
            above the optimum.
            """
        )

        if abs(opt_distance - bf_distance) < 1e-9:
            st.success(
                "For this particular instance, 2-opt reached the same distance as the exact optimum."
            )
    else:
        st.info(
            """
            The exact optimum is not being computed at this problem size, so the
            heuristic optimality gap cannot be calculated.
            """
        )


# ======================================================
# TAB 7 — COMPLEXITY
# ======================================================

with tab7:
    st.header("Why TSP Becomes Computationally Expensive")

    complexity_n = st.slider(
        "Choose number of cities for the complexity demonstration",
        min_value=4,
        max_value=25,
        value=max(5, n),
        key="complexity_n",
    )

    fixed_start_orderings = factorial(complexity_n - 1)

    if is_symmetric(distance_matrix):
        unique_tours = fixed_start_orderings // 2
        formula_text = r"\frac{(n-1)!}{2}"
    else:
        unique_tours = fixed_start_orderings
        formula_text = r"(n-1)!"

    c1, c2 = st.columns(2)
    c1.metric(
        "Orderings with starting city fixed",
        f"{fixed_start_orderings:,}",
    )
    c2.metric(
        "Tours to consider",
        f"{unique_tours:,}",
    )

    st.latex(
        rf"""
        \text{{Number of tours}} = {formula_text}
        """
    )

    tours_per_second = st.number_input(
        "Assumed enumeration speed (tours per second)",
        min_value=1,
        value=1_000_000,
        step=100_000,
    )

    estimated_seconds = unique_tours / tours_per_second

    st.metric(
        "Estimated enumeration time",
        format_seconds(estimated_seconds),
    )

    ns = list(range(4, complexity_n + 1))
    counts = []

    for k in ns:
        count = factorial(k - 1)
        if is_symmetric(distance_matrix):
            count //= 2
        counts.append(count)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ns, counts, marker="o")
    ax.set_yscale("log")
    ax.set_xlabel("Number of cities")
    ax.set_ylabel("Number of tours (log scale)")
    ax.set_title("Growth of the Brute-Force Search Space")
    ax.grid(alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        """
        The log-scale graph still rises sharply. This is why exact enumeration
        becomes impractical very quickly and motivates smarter exact algorithms
        and heuristics.
        """
    )


# ======================================================
# TAB 8 — INTERPRETATION
# ======================================================

with tab8:
    st.header("Interpret the Results")

    st.subheader("Nearest Neighbour")
    st.write(
        f"""
        Starting from **{start_city}**, Nearest Neighbour constructs the route

        **{route_to_text(nn_route, cities)}**

        with total distance **{nn_distance:.3f}**.
        """
    )

    st.subheader("After 2-opt")
    st.write(
        f"""
        2-opt improves the initial heuristic route to

        **{route_to_text(opt_route, cities)}**

        with total distance **{opt_distance:.3f}**.
        """
    )

    if nn_distance > 0:
        st.write(
            f"""
            The local search therefore saves **{nn_distance - opt_distance:.3f}**
            distance units, an improvement of
            **{100 * (nn_distance - opt_distance) / nn_distance:.2f}%**
            relative to the Nearest Neighbour solution.
            """
        )

    if brute_available:
        st.subheader("Comparison with the true optimum")

        st.write(
            f"""
            Brute force finds the exact optimal route

            **{route_to_text(bf_route, cities)}**

            with total distance **{bf_distance:.3f}**.
            """
        )

        if abs(nn_distance - bf_distance) < 1e-9:
            st.success(
                "Nearest Neighbour happened to find an optimal solution for this instance."
            )
        else:
            st.write(
                f"""
                Nearest Neighbour is
                **{100 * (nn_distance - bf_distance) / bf_distance:.2f}%**
                above the optimum.
                """
            )

        if abs(opt_distance - bf_distance) < 1e-9:
            st.success(
                "After 2-opt, the heuristic reached an optimal solution for this instance."
            )
        else:
            st.write(
                f"""
                After 2-opt, the solution remains
                **{100 * (opt_distance - bf_distance) / bf_distance:.2f}%**
                above the optimum.
                """
            )

    st.subheader("Teaching takeaway")

    st.write(
        """
        - **Brute force** provides a benchmark and guarantees optimality, but
          becomes computationally expensive.
        - **Nearest Neighbour** is fast and easy to understand, but is not
          guaranteed to be optimal.
        - **2-opt** can improve a heuristic solution by local edge exchanges,
          but it can stop at a local optimum.
        - Changing the starting city or input distances can change the heuristic
          route, so students can experiment with many instances.
        """
    )
