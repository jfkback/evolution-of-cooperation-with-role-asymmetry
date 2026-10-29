import numpy as np
import numpy.random as random


def clip_value(
    value: float, min_value: float = 0, max_value: float = 1
) -> float:
    return max(min(value, max_value), min_value)


class EvolutionCooperation:

    def __init__(
        self,
        N: int = 1000,
        n: int = 3,
        m: int = 1000,
        beta: float = 1.0,
        sigma: float = 0.05,
        mu: float = 0.01,
        tao: float | None = None,
        eta: float | None = None,
    ) -> None:
        """Main class for the evolution of cooperation simulation.

        Args:
            N (int, optional): The number of agents in the simulation. Defaults
                to 1000.
            n (int, optional): The number of agents in each pool used to
                calculate payoffs. Defaults to 3.
            m (int, optional): The number of rounds in the simulation. Defaults
                to 1000.
            beta (float, optional): Parameter used to control the probability
                that during the update process an agent adopts another agent's
                strategy. Defaults to 1.0.
            sigma (float, optional): The standard deviation of used in the case
                where a mutation occurs for a D-type during the update process.
                Defaults to 0.05.
            mu (float, optional): The probability a D-type will mutate and
                sample from a distribution instead of taking the exact value
                of the other agent. Defaults to 0.01.
            tao (float | None, optional): If provided, this is the probability
                a D-type or R-type will randomly change their strategy. Defaults
                to None.
            eta (float | None, optional): The parameter used in the
                probabilistic payoff calculation. If not provided, the default
                payoff calculation is used which selects the largest payoff.
                Defaults to None.
        """

        self.N = N
        self.n = n
        self.m = m
        self.beta = beta
        self.sigma = sigma
        self.mu = mu
        self.tao = tao
        self.eta = eta
        self.store_agent = None
        self.store_payoff = None

        self.payoff = np.zeros(self.N)
        agent_type = random.uniform(0, 1, self.N)
        agent_type[
            random.choice([False, True], len(agent_type), p=[0.5, 0.5])
        ] = -1.0
        self.agent_type = agent_type

    def probabilistic_payoff_calculation(self, idxs: np.ndarray) -> None:
        """Given an index of agents sampled in one pool, calculate the payoffs
        for the agents in that pool. This is the probabilistic payoff
        calculation where the D-type selected is chosen with probability
        tau_i = exp(p_i / eta) / sum(exp(p_j / eta)).

        If no D-types are present, the payoff is not updated. If no R-types are
        present, the payoff is not updated.

        Args:
            idxs (np.ndarray): The index of the agents sampled in one pool.
        """

        vals = self.agent_type[idxs]

        # Identify D-types (values >= 0) within the chosen group
        # np.where returns a tuple, [0] gets the array of indices relative to 'vals'
        d_local_indices = np.where(vals >= 0)[0]

        # If s=0 (no D-types) or s=n (no R-types), payoff is 0.
        if len(d_local_indices) == 0 or len(d_local_indices) == len(idxs):
            return

        # Get strategies of D-types
        d_strategies = vals[d_local_indices]

        # Calculate Tau probabilities: exp(p_i / eta) / sum(exp(p_j / eta))
        # Note: For very small eta, this can overflow.
        # However, with eta >= 0.01 and p <= 1, exp(100) is safe in float64.
        exponents = np.exp(d_strategies / self.eta)
        sum_exponents = np.sum(exponents)
        probs = exponents / sum_exponents

        # Choose a D-type i at random with probability tau_i
        chosen_local_idx = random.choice(d_local_indices, p=probs)

        best_d_type_idx = idxs[chosen_local_idx]
        best_strategy = vals[chosen_local_idx]  # This is p_hat

        # Calculate payoffs
        # R-types are those with value < 0
        r_type_idxs = idxs[np.where(vals < 0)[0]]

        # We checked earlier that R-types exist (len(d_local_indices) != len(idxs))
        d_payoff = 1 - best_strategy
        r_payoff = best_strategy / len(r_type_idxs)

        self.payoff[r_type_idxs] += r_payoff
        self.payoff[best_d_type_idx] += d_payoff

    def standard_payoff_calculation(self, idxs: np.ndarray) -> None:
        """Given an index of agents sampled in one pool, calculate the payoffs
        for the agents in that pool. This is the standard payoff calculation
        where the D-type with the highest strategy is always chosen.

        If no D-types are present, the payoff is not updated. If no R-types are
        present, the payoff is not updated.

        Args:
            idxs (np.ndarray): The index of the agents sampled in one pool.
        """

        vals = self.agent_type[idxs]
        temp = np.argmax(vals)

        best_d_type_idx = idxs[temp]
        best_strategy = vals[temp]

        if best_strategy < 0:
            return

        r_type_idxs = idxs[np.argwhere(vals < 0)]

        # If there are R-types no payoff is given to the D-types.
        if len(r_type_idxs) > 0:
            d_payoff = 1 - best_strategy
            r_payoff = best_strategy / len(r_type_idxs)

            self.payoff[r_type_idxs] += r_payoff
            self.payoff[best_d_type_idx] += d_payoff

    def payoff_calculation_fast(self, idxs: np.ndarray) -> None:
        """Given an index of agents sampled in one pool, calculate the payoffs
        for the agents in that pool. Either the probabilistic or standard payoff
        calculation is used based on whether eta is set or not.

        Args:
            idxs (np.ndarray): The index of the agents sampled in one pool.
        """

        if self.eta is not None:
            self.probabilistic_payoff_calculation(idxs)
        else:
            self.standard_payoff_calculation(idxs)

    def multipayoff_calculation(self) -> None:
        """Repeatedly runs the payoff calculation with a random sample of n items
        from the population of N agents, m times. The payoff is updated in place
        for each agent in the sample.
        """

        for _ in range(self.m):
            self.payoff_calculation_fast(random.choice(self.N, self.n))

    def prob_replace(self, payoff_i: float, payoff_j: float) -> bool:
        """Given the payoff of agent i and agent j this function produces
        a boolean value to indicate whether agent i should adopt the
        strategy of agent j.

        Args:
            payoff_i (float): The payoff of agent i.
            payoff_j (float): The payoff of agent j.

        Returns:
            bool: Whether agent i should adopt the strategy of agent j.
        """

        p = 1 / (1 + np.e ** (-self.beta * (payoff_j - payoff_i)))
        return p > random.uniform(size=1)[0]

    def update_agents(self) -> None:
        """Randomly selects two agents i and j from the population of N agents.
        Based on their payoffs, it determines whether agent i should adopt the
        strategy of agent j.

        The agent types are updated in place based on the following rules:
        - If agent j is a D-type and a random number is less than mu, agent
            i adopts a new strategy drawn from a normal distribution centered
            at agent j's strategy.
        - If tao is set and a random number is less than tao, agent i adopts the
            opposite type of agent j. If agent j is a R-type, agent i adopts a
            new strategy drawn from a uniform distribution. If agent j is a
            D-type, agent i becomes a R-type.
        - In the default case, agent i adopts the type of agent j.
        """

        i, j = random.randint(0, self.N, size=2)

        # Based on the payoffs of agents i and j the probability of replacement
        # is calculated, and if the probability is greater than a random number
        # between 0 and 1, the replacement process occurs.
        if self.prob_replace(self.payoff[i], self.payoff[j]):
            # If agent j is a D-type (agent_type[j] != -1) and a random number
            # is less than mu, the strategy of agent i is updated to a new
            # strategy drawn from a normal distribution centered at agent j's
            # strategy with standard deviation sigma.
            if (
                self.agent_type[j] != -1
                and self.mu > random.uniform(size=1)[0]
            ):
                self.agent_type[i] = clip_value(
                    random.normal(self.agent_type[j], self.sigma, 1)[0]
                )

            # If tao is set, and the probability of replacement is greater
            # than a random number, then agent i will adopt the opposite type
            # of agent j. If agent j is a R-type (agent_type[j] == -1),
            # agent i will adopt a new strategy drawn from a uniform
            # distribution between 0 and 1. If agent j is a D-type, agent i
            # will become a R-type.
            elif self.tao is not None and self.tao > random.uniform(size=1)[0]:
                if self.agent_type[j] == -1:
                    self.agent_type[i] = random.uniform(size=1)[0]
                else:
                    self.agent_type[i] = -1

            # In the default case, agent i adopts the type of agent j.
            else:
                self.agent_type[i] = self.agent_type[j]

    def multi_update_agents(self) -> None:
        """Run `self.update_agents` N times. This is a single "reproduction"
        stage.
        """
        for _ in range(self.N):
            self.update_agents()

    def run(self, num_runs: int) -> None:
        """Runs the complete process on the full population with `num_runs`
        number of generations. At each generation, the payoffs are recalculated
        and the agents are updated based on their payoffs.

        Both the agent type and the payoff are stored at each generation in
        `self.store_agent` and `self.store_payoff` respectively.

        Args:
            num_runs (int): The number of generations to use.
        """

        random.seed()
        self.store_agent = np.zeros((num_runs, self.N))
        self.store_payoff = np.zeros((num_runs, self.N))

        for i in range(num_runs):
            num_d = (self.agent_type >= 0).sum()
            num_r = (self.agent_type < 0).sum()

            if num_d == 0 or num_r == 0:
                return

            self.store_agent[i] = self.agent_type
            self.multipayoff_calculation()
            self.store_payoff[i] = self.payoff
            self.multi_update_agents()
            self.payoff = np.zeros(self.N)

    def save_as_csv(
        self, agent_type_path: str, payoff_path: str, suffix: str = None
    ) -> None:
        """Saves the `self.storage_agent` and `self.store_payoff` output as
        CSV files.

        Args:
            agent_type_path (str): The directory where the agent type file
                will be saved.
            payoff_path (str): The directory where the payoff file will be
                saved.
            suffix (str, optional): A suffix to append to the filenames.
                Defaults to None.
        """

        eta_str = f"_{self.eta}" if self.eta is not None else ""
        if suffix is not None:
            np.savetxt(
                f"{agent_type_path}/{self.n}_{self.m}_{self.beta}_{self.sigma}_{self.mu}{eta_str}_agent_type_{suffix}.csv",
                self.store_agent,
                fmt="%1.6f",
                delimiter=",",
            )
            np.savetxt(
                f"{payoff_path}/{self.n}_{self.m}_{self.beta}_{self.sigma}_{self.mu}{eta_str}_payoff_{suffix}.csv",
                self.store_payoff,
                fmt="%1.6f",
                delimiter=",",
            )
        else:
            np.savetxt(
                f"{agent_type_path}/agent_type.csv",
                self.store_agent,
                delimiter=",",
            )
            np.savetxt(
                f"{payoff_path}/payoff.csv", self.store_payoff, delimiter=","
            )

    def test_one(self):
        self.agent_type = np.full(2, -1)
        self.payoff = np.zeros(2)
        self.payoff_calculation_fast(np.arange(0, 2))
        assert (
            self.payoff[0] == 0 and self.payoff[1] == 0
        ), "All R-Type Payoff Incorrect"

        self.agent_type = np.full(2, 0.5)
        self.payoff = np.zeros(2)
        self.payoff_calculation_fast(np.arange(0, 2))
        assert (
            self.payoff[0] == 0 and self.payoff[1] == 0
        ), "All D-Type Payoff Incorrect"

        self.agent_type = np.array([-1, 0.5])
        self.payoff = np.zeros(2)
        self.payoff_calculation_fast(np.arange(0, 2))
        assert (
            self.payoff[0] == 0.5 and self.payoff[1] == 0.5
        ), "One D-Type, One R-Type Payoff Incorrect"

        self.agent_type = np.array([-1, 0.7])
        self.payoff = np.zeros(2)
        self.payoff_calculation_fast(np.arange(0, 2))
        assert (
            self.payoff[0] == 0.7 and round(self.payoff[1], 1) == 0.3
        ), "One D-Type, One R-Type Payoff Incorrect"

        self.agent_type = np.array([-1, 0.7, 0.4])
        self.payoff = np.zeros(3)
        self.payoff_calculation_fast(np.arange(0, 3))
        assert (
            self.payoff[0] == 0.7
            and round(self.payoff[1], 1) == 0.3
            and round(self.payoff[2], 1) == 0.0
        ), "Two D-Type, One R-Type Payoff Incorrect"

        self.agent_type = np.array([-1, -1, 0.4])
        self.payoff = np.zeros(3)
        self.payoff_calculation_fast(np.arange(0, 3))
        assert (
            self.payoff[0] == 0.2
            and round(self.payoff[1], 1) == 0.2
            and round(self.payoff[2], 1) == 0.6
        ), "One D-Type, Two R-Type Payoff Incorrect"

    def test_two(self):
        # i=2, j=1 P_(i <--- j) = .268941
        count_true, total = 0, 0
        for _ in range(10000):
            if self.prob_replace(2, 1):
                count_true += 1
            total += 1
        assert (
            abs((count_true / total) - 0.2689414213699951) < 0.01
        ), "Payoff percentage off"

        # i=1, j=2 P_(i <--- j) = 0.731058
        count_true, total = 0, 0
        for _ in range(10000):
            if self.prob_replace(1, 2):
                count_true += 1
            total += 1
        assert (
            abs((count_true / total) - 0.7310585786300049) < 0.01
        ), "Payoff percentage off"


def get_average_p(agent_type: np.ndarray) -> np.ndarray:
    cycles, _ = agent_type.shape
    ret = np.zeros(cycles)
    for i in range(cycles):
        curr = agent_type[i]
        curr = curr[curr != -1]
        ret[i] = np.average(curr)
    return ret


def get_num_d(agent_type: np.ndarray) -> np.ndarray:
    cycles, _ = agent_type.shape
    ret = np.zeros(cycles)
    for i in range(cycles):
        curr = agent_type[i]
        curr = curr[curr != -1]
        ret[i] = np.average(curr)
    return ret


def one_run(
    num_runs: int,
    i: int,
    basepath: str,
    N: int = 1000,
    n: int = 3,
    m: int = 1000,
    beta: float = 1.0,
    sigma: float = 0.05,
    mu: float = 0.01,
    eta: float | None = None,
    tao: float | None = None,
) -> None:
    """Run a single iteration and saves the output as a CSV."""
    random.seed()
    model = EvolutionCooperation(N, n, m, beta, sigma, mu, tao=tao, eta=eta)
    model.run(num_runs)
    model.save_as_csv(f"{basepath}/Agent Type", f"{basepath}/Payoff", f"{i}")
