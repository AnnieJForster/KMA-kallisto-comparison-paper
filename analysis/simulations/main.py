"""Run the six simulation scenarios described in the paper."""
import argparse
import json
import os
import shutil

from .process import (
    run_kmer_simulation_RA_D,
    run_kmer_simulation_ID_M,
    run_kmer_simulation_ID_D,
    run_kmer_simulation_RA_prop,
    run_kmer_simulation_RA_deltaM,
    run_kmer_simulation_ID_DM,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--simulation-sets", type=int, default=10)
    parser.add_argument("--seq-length", type=int, default=10000)
    parser.add_argument("--num-reads", type=int, default=1000)
    parser.add_argument("--mutation-counts", type=int, nargs="+")
    parser.add_argument("--proportion-mutations", type=int, default=250)
    parser.add_argument("--experiments", nargs="+", choices=[
        "RA_D", "ID_M", "ID_D", "RA_prop", "RA_deltaM", "ID_DM"
    ], default=["RA_D", "ID_M", "ID_D", "RA_prop", "RA_deltaM", "ID_DM"])
    args = parser.parse_args()
    mutation_counts = args.mutation_counts
    if mutation_counts is None:
        mutation_counts = (list(range(1, 101)) + list(range(105, 251, 5))
                           + list(range(260, 501, 10)) + list(range(550, 1001, 50))
                           + list(range(1100, 2001, 100)))
    if min(args.seq_length, args.num_reads, args.simulation_sets) <= 0:
        parser.error("Sequence length, read count, and simulation sets must be positive")
    if min(mutation_counts) < 0 or args.proportion_mutations < 0:
        parser.error("Mutation counts must be nonnegative")
    # Existing helpers reuse intermediate files. A new directory prevents mixing runs.
    if os.path.exists(args.output_dir):
        parser.error("Choose a new output directory for each simulation run")
    missing = [tool for tool in ["kallisto", "kma", "wgsim"] if shutil.which(tool) is None]
    if missing:
        parser.error("Missing programs on PATH: " + ", ".join(missing))
    os.makedirs(args.output_dir)
    parameters = vars(args).copy()
    parameters["mutation_counts"] = mutation_counts
    with open(os.path.join(args.output_dir, "simulation_parameters.json"), "w") as f:
        json.dump(parameters, f, indent=2)
    for i in range(args.simulation_sets):
        output_dir = os.path.join(args.output_dir, f"simulation_set_{i + 1}")
        for experiment in args.experiments:
            print(f"Simulation set {i + 1}: {experiment}", flush=True)
            if experiment == "RA_prop":
                run_kmer_simulation_RA_prop(
                    output_dir, seq_length=args.seq_length, num_reads=args.num_reads,
                    mutation_count=args.proportion_mutations)
            else:
                functions = {
                    "RA_D": run_kmer_simulation_RA_D,
                    "ID_M": run_kmer_simulation_ID_M,
                    "ID_D": run_kmer_simulation_ID_D,
                    "RA_deltaM": run_kmer_simulation_RA_deltaM,
                    "ID_DM": run_kmer_simulation_ID_DM,
                }
                functions[experiment](output_dir, seq_length=args.seq_length,
                                      num_reads=args.num_reads, mutation_counts=mutation_counts)


if __name__ == "__main__":
    main()
