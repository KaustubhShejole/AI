import random
goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]


def count_inversions(state):
    inversions = 0
    for i in range(len(state)):
        for j in range(i + 1, len(state)):
            if state[i] > state[j] and state[i] != 0 and state[j] != 0:
                inversions += 1
    return inversions


inversions_goal_state = count_inversions(goal_state)


def is_solvable(state, goal_state):
    inversions_initial_state = count_inversions(state)
    return (inversions_initial_state % 2 == inversions_goal_state % 2)


def generate_random_state():
    state = list(range(9))
    random.shuffle(state)
    return state


def find_solvable_initial_state(goal_state):
    while True:
        initial_state = generate_random_state()
        if is_solvable(initial_state, goal_state) and initial_state != goal_state:
            return initial_state


def is_goal_state(state):
    return state == [1, 2, 3, 4, 5, 6, 7, 8, 0]


def apply_action(state, action):
    new_state = list(state)
    blank_index = new_state.index(0)
    new_index = blank_index + action
    new_state[blank_index], new_state[new_index] = new_state[new_index], new_state[blank_index]
    return new_state


def get_actions(state):
    actions = []
    blank_index = state.index(0)
    if blank_index - 3 >= 0:  # Up
        actions.append(-3)
    if blank_index + 3 < 9:  # Down
        actions.append(3)
    if blank_index % 3 > 0:  # Left
        actions.append(-1)
    if blank_index % 3 < 2:  # Right
        actions.append(1)
    return actions


def iddfs_optimized(initial_state):
    depth_limit = 0
    number_of_duplicates = 0
    while True:
        expanded_nodes = set()
        solution_path, actions, number_of_duplicates = depth_limited_dfs_optimized(
            initial_state, depth_limit, expanded_nodes, number_of_duplicates)
        if solution_path:
            print("Number of expanded nodes stored = "+str(len(expanded_nodes)))
            print("Number of duplicate nodes = " + str(number_of_duplicates))
            return solution_path, actions
        print(depth_limit)
        depth_limit += 1


def depth_limited_dfs_optimized(state, depth_limit, expanded_nodes, number_of_duplicates):
    if is_goal_state(state):
        return [state], [], number_of_duplicates

    if depth_limit == 0:
        return None, [], number_of_duplicates

    expanded_nodes.add((tuple(state), depth_limit))

    for action in get_actions(state):
        child_state = apply_action(state, action)
        if is_goal_state(child_state):
            return [state, child_state], [action], number_of_duplicates
        if (tuple(child_state), depth_limit - 1) not in expanded_nodes:
            result, actions, number_of_duplicates = depth_limited_dfs_optimized(
                child_state, depth_limit - 1, expanded_nodes, number_of_duplicates)
            if result is not None:
                return [state] + result, [action] + actions, number_of_duplicates
        else:
            number_of_duplicates = number_of_duplicates + 1

    return None, [], number_of_duplicates


# Example initial state
initial_state = find_solvable_initial_state(goal_state)
print("Initial state = "+str(initial_state))

# Run IDDFS algorithm and print the solution path and actions
solution_path, actions = iddfs_optimized(initial_state)
if solution_path:
    print("Solution Path:", solution_path)
    print("Solution Actions:", actions)
else:
    print("No solution found.")
