import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, gaussian_kde
import sys
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functionalANOVA.two_group_cov import two_group_cov
from functionalANOVA.k_group_cov import k_group_cov
from functionalANOVA.k_group_cov_pairwise import k_group_cov_pairwise


class MockFANOVA:
    def __init__(self, n_simul, n_boot, n_permutations, k_groups, n_i):
        self.N_simul = n_simul
        self.N_boot = n_boot
        self.N_permutations = n_permutations
        self.k_groups = k_groups
        default_samples = 50
        self.n_i = n_i if n_i else [default_samples] * k_groups
        self.N = sum(self.n_i)
        self.n_domain_points = 10

    @staticmethod
    def chi_sq_mixture(df, coefs, N_samples):
        n_eigs = len(coefs)
        chi2rvs = np.random.chisquare(df, size=(n_eigs, N_samples))
        T_null = (coefs @ chi2rvs)
        return T_null

    @staticmethod
    def aflag_maker(n_i):
        aflag = []
        for k in range(len(n_i)):
            indicator = np.repeat(k + 1, n_i[k])
            aflag.extend(indicator)
        return np.array(aflag)


def generate_data(scedasticity, k_groups, n_samples, p_vars, seed):
    np.random.seed(seed)

    default_samples = 50
    if n_samples is None:
        n_samples = [default_samples] * k_groups
    elif isinstance(n_samples, int):
        n_samples = [n_samples] * k_groups

    data_groups = []
    true_covs = []

    if scedasticity == 'homo':
        # homoscedastic: all groups have same covariance
        diagonal_scale = 0.5
        off_diagonal_scale = 0.1
        base_cov = np.eye(p_vars) * diagonal_scale + np.ones((p_vars, p_vars)) * off_diagonal_scale
        for i in range(k_groups):
            data = multivariate_normal.rvs(mean=np.zeros(p_vars), cov=base_cov, size=n_samples[i])
            data_groups.append(data)
            true_covs.append(base_cov)

    elif scedasticity == 'hetero':
        for i in range(k_groups):
            base_scale = 0.2
            scale_increment = 0.4
            base_correlation = 0.1
            correlation_increment = 0.1

            scale = base_scale + i * scale_increment
            correlation = base_correlation + i * correlation_increment
            cov = np.eye(p_vars) * scale + np.ones((p_vars, p_vars)) * correlation
            data = multivariate_normal.rvs(mean=np.zeros(p_vars), cov=cov, size=n_samples[i])
            data_groups.append(data)
            true_covs.append(cov)

    return data_groups, true_covs


def run_two_group_test(method, data_groups, fanova):
    """run two-group covariance test"""
    required_groups = 2
    if len(data_groups) != required_groups:
        raise ValueError("two-group test requires exactly 2 groups")

    y1, y2 = data_groups[0], data_groups[1]
    pvalue = two_group_cov(fanova, method, y1, y2)
    return pvalue


def run_k_group_test(method, data_groups, fanova):
    """run k-group covariance test"""
    # prepare data for k_group_cov
    data_transposed = [group.T for group in data_groups]
    fanova.data = data_transposed

    # compute pooled statistics
    all_data = np.vstack(data_groups)
    V = all_data - np.mean(all_data, axis=0)

    # compute test statistic
    stat = 0
    pooled_cov = np.cov(all_data.T)

    for i, group in enumerate(data_groups):
        group_cov = np.cov(group.T)
        stat += (len(group) - 1) * np.trace((group_cov - pooled_cov) @ (group_cov - pooled_cov))

    pvalue = k_group_cov(fanova, method, stat, pooled_cov, V)
    return pvalue


def run_pairwise_test(method, data_groups, fanova):
    """run pairwise covariance tests"""
    min_groups = 2
    if len(data_groups) < min_groups:
        raise ValueError("pairwise test requires at least 2 groups")

    pvalues = {}
    for i in range(len(data_groups)):
        for j in range(i + 1, len(data_groups)):
            y1, y2 = data_groups[i], data_groups[j]
            pvalue = k_group_cov_pairwise(fanova, method, y1, y2)
            pvalues[f"group_{i+1}_vs_group_{j+1}"] = pvalue

    return pvalues


def visualize_results(data_groups, true_covs, results, test_type, scedasticity, output_path):
    """create comprehensive visualization of results"""
    k_groups = len(data_groups)

    if test_type == 'pairwise':
        # for pairwise, just show first comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        pair_name = list(results.keys())[0]
        pvalue = results[pair_name]
        fig.suptitle(f'{scedasticity} data - pairwise test: {pair_name} (p={pvalue})', fontsize=16)

        y1, y2 = data_groups[0], data_groups[1]
        cov1, cov2 = np.cov(y1.T), np.cov(y2.T)

    else:
        if k_groups == 2:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            y1, y2 = data_groups[0], data_groups[1]
            cov1, cov2 = np.cov(y1.T), np.cov(y2.T)

            if isinstance(results, dict):
                result_str = ", ".join([f"{k}: {v}" for k, v in results.items()])
            else:
                result_str = f"p={results}"

            fig.suptitle(f'{scedasticity} data - {test_type} test ({result_str})', fontsize=16)
        else:
            fig, axes = plt.subplots(2, min(k_groups, 3), figsize=(18, 12))
            if isinstance(results, dict):
                result_str = ", ".join([f"{k}: {v}" for k, v in results.items()])
            else:
                result_str = f"p={results}"
            fig.suptitle(f'{scedasticity} data - {test_type} test with {k_groups} groups ({result_str})', fontsize=16)

    # plot for 2-group case
    if k_groups >= 2:
        y1, y2 = data_groups[0], data_groups[1]
        cov1, cov2 = np.cov(y1.T), np.cov(y2.T)
        cov_diff = cov1 - cov2

        # scatter plot
        axes[0, 0].scatter(y1[:, 0], y1[:, 1], alpha=0.6, label='group 1', color='blue')
        axes[0, 0].scatter(y2[:, 0], y2[:, 1], alpha=0.6, label='group 2', color='red')
        axes[0, 0].set_xlabel('variable 1')
        axes[0, 0].set_ylabel('variable 2')
        axes[0, 0].set_title('scatter plot of first two variables')
        axes[0, 0].legend()

        # covariance matrices
        im1 = axes[0, 1].imshow(cov1, cmap='viridis')
        axes[0, 1].set_title('group 1 covariance matrix')
        plt.colorbar(im1, ax=axes[0, 1])

        im2 = axes[0, 2].imshow(cov2, cmap='viridis')
        axes[0, 2].set_title('group 2 covariance matrix')
        plt.colorbar(im2, ax=axes[0, 2])

        # covariance difference
        im3 = axes[1, 0].imshow(cov_diff, cmap='RdBu', vmin=-np.abs(cov_diff).max(), vmax=np.abs(cov_diff).max())
        axes[1, 0].set_title('covariance difference (group1 - group2)')
        plt.colorbar(im3, ax=axes[1, 0])

        # density plots
        data_combined = np.vstack([y1[:, 0], y2[:, 0]])
        kde1 = gaussian_kde(y1[:, 0])
        kde2 = gaussian_kde(y2[:, 0])
        x_range = np.linspace(data_combined.min(), data_combined.max(), 200)

        axes[1, 1].plot(x_range, kde1(x_range), label='group 1 density', color='blue')
        axes[1, 1].plot(x_range, kde2(x_range), label='group 2 density', color='red')
        axes[1, 1].set_xlabel('variable 1 values')
        axes[1, 1].set_ylabel('density')
        axes[1, 1].set_title('density estimates for variable 1')
        axes[1, 1].legend()

        # results summary
        if isinstance(results, dict):
            methods = list(results.keys())
            pvalues = list(results.values())
            methods_short = [m.replace('-', '\n') for m in methods]
            bars = axes[1, 2].bar(methods_short, pvalues)
            alpha_threshold = 0.05
            axes[1, 2].axhline(y=alpha_threshold, color='red', linestyle='--', label='α = 0.05')
            axes[1, 2].set_ylabel('p-value')
            axes[1, 2].set_title('p-values by method')
            axes[1, 2].legend()
            axes[1, 2].tick_params(axis='x', rotation=45)
        else:
            axes[1, 2].text(0.5, 0.5, f'p-value: {results}',
                          horizontalalignment='center', verticalalignment='center',
                          transform=axes[1, 2].transAxes, fontsize=14)
            axes[1, 2].set_title('test result')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='general covariance testing tool')

    # data generation options
    parser.add_argument('--scedasticity', choices=['homo', 'hetero'], default='homo',
                       help='type of data to generate (default: homo)')
    parser.add_argument('--groups', type=int, default=2,
                       help='number of groups to generate (default: 2)')
    parser.add_argument('--samples', type=int, nargs='+', default=None,
                       help='sample sizes per group (default: 50 for each)')
    parser.add_argument('--vars', type=int, default=10,
                       help='number of variables/dimensions (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='random seed (default: 42)')

    # test options
    parser.add_argument('--test', choices=['two-group', 'k-group', 'pairwise'], default='two-group',
                       help='type of covariance test (default: two-group)')
    parser.add_argument('--method', choices=['L2-Simul', 'L2-Naive', 'L2-BiasReduced', 'Bootstrap-Test', 'Permutation-Test'],
                       nargs='+', default=['L2-Simul'],
                       help='test method(s) to use (default: L2-Simul)')

    # simulation options
    parser.add_argument('--n-simul', type=int, default=1000,
                       help='number of simulations (default: 1000)')
    parser.add_argument('--n-boot', type=int, default=100,
                       help='number of bootstrap samples (default: 100)')
    parser.add_argument('--n-perm', type=int, default=100,
                       help='number of permutations (default: 100)')

    # output options
    parser.add_argument('--output', type=str, default=None,
                       help='output plot filename (default: auto-generated)')
    parser.add_argument('--verbose', action='store_true',
                       help='verbose output')

    args = parser.parse_args()

    # validate arguments
    if args.test == 'two-group' and args.groups != 2:
        parser.error("two-group test requires exactly 2 groups")
    if args.test == 'k-group' and args.groups < 2:
        parser.error("k-group test requires at least 2 groups")
    if args.test == 'pairwise' and args.groups < 2:
        parser.error("pairwise test requires at least 2 groups")

    # generate data
    console = Console()

    if args.verbose:
        default_samples = 50
        console.print(f"generating {args.scedasticity}scedastic data...")
        console.print(f"  groups: {args.groups}")
        console.print(f"  samples per group: {args.samples or [default_samples] * args.groups}")
        console.print(f"  variables: {args.vars}")
        console.print(f"  seed: {args.seed}")

    data_groups, true_covs = generate_data(
        scedasticity=args.scedasticity,
        k_groups=args.groups,
        n_samples=args.samples,
        p_vars=args.vars,
        seed=args.seed
    )

    # create mock fanova object
    fanova = MockFANOVA(
        n_simul=args.n_simul,
        n_boot=args.n_boot,
        n_permutations=args.n_perm,
        k_groups=args.groups,
        n_i=[len(group) for group in data_groups]
    )

    # run tests based on command line arguments
    if args.verbose:
        console.print(f"\nrunning {args.test} test with method(s): {args.method}")

    # determine test type
    if args.test == 'two-group':
        if len(args.method) == 1:
            method = args.method[0]
            result = run_two_group_test(method, data_groups, fanova)
            if args.verbose:
                console.print(f"  p-value = {result}")
        else:
            result = {}
            for method in args.method:
                pval = run_two_group_test(method, data_groups, fanova)
                result[method] = pval
                if args.verbose:
                    console.print(f"  {method}: p-value = {pval}")

    elif args.test == 'k-group':
        if len(args.method) == 1:
            method = args.method[0]
            result = run_k_group_test(method, data_groups, fanova)
            if args.verbose:
                console.print(f"  p-value = {result}")
        else:
            result = {}
            for method in args.method:
                pval = run_k_group_test(method, data_groups, fanova)
                result[method] = pval
                if args.verbose:
                    console.print(f"  {method}: p-value = {pval}")

    elif args.test == 'pairwise':
        if len(args.method) == 1:
            method = args.method[0]
            result = run_pairwise_test(method, data_groups, fanova)
            if args.verbose:
                for pair, pval in result.items():
                    console.print(f"  {pair}: p-value = {pval}")
        else:
            result = {}
            for method in args.method:
                pval_dict = run_pairwise_test(method, data_groups, fanova)
                pval = list(pval_dict.values())[0]
                result[method] = pval
                if args.verbose:
                    console.print(f"  {method}: p-value = {pval}")

    # generate output filename
    if args.output is None:
        args.output = f"/Users/shankarchawla/math/projects/fanova/F-ANOVA-py/tests/{args.scedasticity}_{args.test}_{args.groups}groups.png"

    # create visualization
    visualize_results(data_groups, true_covs, result, args.test, args.scedasticity, args.output)

    if args.verbose:
        console.print(f"\nvisualization saved to: {args.output}")

    # summary with rich formatting
    alpha_threshold = 0.05

    table = Table(title=f"{args.scedasticity.title()}scedastic {args.test.title()} Test Results")
    table.add_column("Method/Comparison", style="cyan")
    table.add_column("P-value", style="magenta")
    table.add_column("Decision", style="green")

    if isinstance(result, dict):
        for method, pval in result.items():
            if not np.isnan(pval):
                if pval < alpha_threshold:
                    status = "Reject H0"
                    status_style = "bold red"
                else:
                    status = "Fail to Reject H0"
                    status_style = "bold green"

                table.add_row(method, str(pval), Text(status, style=status_style))
    else:
        if isinstance(result, dict):
            for pair, pval in result.items():
                if pval < alpha_threshold:
                    status = "Reject H0"
                    status_style = "bold red"
                else:
                    status = "Fail to Reject H0"
                    status_style = "bold green"

                table.add_row(pair, str(pval), Text(status, style=status_style))
        else:
            if result < alpha_threshold:
                status = "Reject H0"
                status_style = "bold red"
            else:
                status = "Fail to Reject H0"
                status_style = "bold green"

            table.add_row("Test Result", str(result), Text(status, style=status_style))

    console.print(table)


if __name__ == "__main__":
    main()
