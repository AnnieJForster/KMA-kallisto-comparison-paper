import subprocess
from Bio import SeqIO
import polars as pl
import os
import shutil
import shlex
import tempfile
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def mutate_sequence(seq: str, num_mutations: int) -> str:
    seq_list = list(seq)
    for _ in range(num_mutations):
        pos = np.random.randint(len(seq_list))
        original_base = seq_list[pos]
        bases = ["A", "C", "G", "T"]
        bases.remove(original_base)
        new_base = np.random.choice(bases)
        seq_list[pos] = new_base
    return "".join(seq_list)

def build_kallisto_index(fasta_file: str, index_prefix: str) -> None:
    command = f"kallisto index -i {index_prefix} {fasta_file}"
    subprocess.run(command, shell=True, check=True)

def build_kma_index(fasta_file: str, index_prefix: str) -> None:
    command = f"kma index -i {fasta_file} -o {index_prefix}"
    subprocess.run(command, shell=True, check=True)

def run_kallisto_quantification(index_prefix: str, read1: str, read2: str, output_dir: str) -> None:
    # Kallisto fails if output_dir already exists from a previous run.
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if not os.path.exists(index_prefix):
        raise FileNotFoundError(f"Kallisto index not found: {index_prefix}")
    if not os.path.exists(read1):
        raise FileNotFoundError(f"Read1 not found: {read1}")
    if not os.path.exists(read2):
        raise FileNotFoundError(f"Read2 not found: {read2}")

    command = (
        f"kallisto quant -i {shlex.quote(index_prefix)} "
        f"-o {shlex.quote(output_dir)} {shlex.quote(read1)} {shlex.quote(read2)}"
    )
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        combined_output = f"{result.stdout}\n{result.stderr}".lower()
        if "zero reads pseudoaligned" in combined_output:
            os.makedirs(output_dir, exist_ok=True)
            abundance_file = os.path.join(output_dir, "abundance.tsv")
            # Header-only file so downstream parsing yields an empty dataframe with expected columns.
            with open(abundance_file, "w") as f:
                f.write("target_id\tlength\teff_length\test_counts\ttpm\n")
            return
        raise RuntimeError(
            "kallisto quant failed\n"
            f"index: {index_prefix}\n"
            f"read1: {read1}\n"
            f"read2: {read2}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

def run_kma_quantification(index_prefix: str, read1: str, read2: str, output_prefix: str) -> None:
    command = f"kma -ipe {read1} {read2} -o {output_prefix} -t_db {index_prefix} -t 4 -ef -ex_mode -vcf 2 -1t1 > /dev/null" # check this matches
    subprocess.run(command, shell=True, check=True)
    # delete the frag file to save space
    if os.path.exists(f"{output_prefix}.frag.gz"):
        os.remove(f"{output_prefix}.frag.gz")
    # and vcf file
    if os.path.exists(f"{output_prefix}.vcf.gz"):
        os.remove(f"{output_prefix}.vcf.gz")

def simulate_reads_with_wgsim(fasta_file: str, output_prefix: str, num_reads: int=1000, read_length: int=100, error_rate: float=0.01) -> None:
    command = f"wgsim -N {num_reads} -1 {read_length} -2 {read_length} -e {error_rate} {fasta_file} {output_prefix}_R1.fastq {output_prefix}_R2.fastq"
    subprocess.run(command, shell=True, check=True)

def mix_reads_in_equal_proportions(read1_a: str, read2_a: str, read1_b: str, read2_b: str, output_read1: str, output_read2: str) -> None:
    command_mix_read1 = f"cat {read1_a} {read1_b} > {output_read1}"
    command_mix_read2 = f"cat {read2_a} {read2_b} > {output_read2}"
    subprocess.run(command_mix_read1, shell=True, check=True)
    subprocess.run(command_mix_read2, shell=True, check=True)

def return_kmer_metrics(fasta1: str, fasta2: str, kmer_size: int) -> tuple:
    # open fasta files and get sequences
    seq1 = str(next(SeqIO.parse(fasta1, "fasta")).seq)
    seq2 = str(next(SeqIO.parse(fasta2, "fasta")).seq)
    # get kmers
    kmers1 = set(seq1[i:i+kmer_size] for i in range(len(seq1) - kmer_size + 1))
    kmers2 = set(seq2[i:i+kmer_size] for i in range(len(seq2) - kmer_size + 1))
    # compute metrics
    n_shared_kmers = len(kmers1.intersection(kmers2))
    n_total_kmers = len(kmers1.union(kmers2))
    n_kmers_unique_to_1 = len(kmers1.difference(kmers2))
    n_kmers_unique_to_2 = len(kmers2.difference(kmers1))
    return n_shared_kmers, n_total_kmers, n_kmers_unique_to_1, n_kmers_unique_to_2

def parse_kallisto_output(abundance_file: str) -> pl.DataFrame:
    df = pl.read_csv(abundance_file, separator="\t")
    df = df.select([
        pl.col("target_id").alias("reference"),
        pl.col("est_counts").alias("kallisto_estimated_counts"),
        pl.col("tpm").alias("kallisto_tpm"),
    ])
    # Robust to empty/header-only files and mixed inference across runs.
    df = df.with_columns(
        pl.col("kallisto_estimated_counts").cast(pl.Float64, strict=False).fill_null(0.0).alias("kallisto_estimated_counts"),
        pl.col("kallisto_tpm").cast(pl.Float64, strict=False).fill_null(0.0).alias("kallisto_tpm"),
    )
    return df

def parse_kma_output(kma_output_file: str) -> pl.DataFrame:
    df = pl.read_csv(kma_output_file, separator="\t", comment_prefix="##")
    df = df.select([
        pl.col("# refSequence").alias("reference"),
        pl.col("readCount").alias("kma_read_count"),
        pl.col("fragmentCount").alias("kma_fragment_count"),
    ])
    return df

def compute_log_error(expected_A: int, expected_B: int, observed_A: int, observed_B: int, epsilon: float = 1e-10) -> float:
    # negative when biased to A, positive when biased to B
    expected_ratio = expected_A / expected_B if expected_B > 0 else 0.0
    observed_ratio = observed_A / observed_B if observed_B > 0 else 0.0
    return np.log10(observed_ratio + epsilon) - np.log10(expected_ratio + epsilon)

def compute_distinguishibility_D(n_shared_kmers: int, n_kmers_unique_to_1: int, n_kmers_unique_to_2: int) -> float:
    numerator = n_kmers_unique_to_1 + n_kmers_unique_to_2
    denominator = n_shared_kmers + n_kmers_unique_to_1 + n_kmers_unique_to_2
    return numerator / denominator if denominator > 0 else 0.0

def return_kallisto_read_count_for_reference(kallisto_df: pl.DataFrame, reference: str) -> float:
    row = kallisto_df.filter(pl.col("reference") == reference)
    if row.shape[0] == 0:
        return 0.0
    value = row["kallisto_estimated_counts"].cast(pl.Float64, strict=False).fill_null(0.0).item()
    return float(value)

def return_kma_read_count_for_reference(kma_df: pl.DataFrame, reference: str) -> float:
    row = kma_df.filter(pl.col("reference") == reference)
    return float(row["kma_read_count"].item())/2 if row.shape[0] > 0 else 0.0

def compute_matching_M(fasta1: str, fasta2: str, kmer_size: int) -> float:
    # fraction of kmers in 1 that are present in 2
    seq1 = str(next(SeqIO.parse(fasta1, "fasta")).seq)
    seq2 = str(next(SeqIO.parse(fasta2, "fasta")).seq)
    kmers1 = set(seq1[i:i+kmer_size] for i in range(len(seq1) - kmer_size + 1))
    kmers2 = set(seq2[i:i+kmer_size] for i in range(len(seq2) - kmer_size + 1))
    n_shared = len(kmers1.intersection(kmers2))
    n_total_kmers1 = len(kmers1)
    return n_shared / n_total_kmers1 if n_total_kmers1 > 0 else 0.0

def generate_references_and_reads_for_simulation(output_dir: str, seq_length: int, num_reads: int, mutation_counts: list) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "references"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "reads"), exist_ok=True)
    # if already exists, skip this step to save time, as it can be time consuming to generate reads and build indexes
    if not os.path.exists(os.path.join(output_dir, "references", f"original_simulated_sequence.length={seq_length}.fasta")):
        # step 1: simulate a random 1kb sequence with 50% GC content
        simulated_seq = "".join(np.random.choice(list("ACGT"), size=seq_length, p=[0.25, 0.25, 0.25, 0.25]))
        # write to a fasta file
        fasta_file = os.path.join(output_dir, "references", f"original_simulated_sequence.length={seq_length}.fasta")
        with open(fasta_file, "w") as f:
            f.write(f">simulated_sequence.length={seq_length}\n{simulated_seq}\n")
        # simulate reads from the original sequence
        simulate_reads_with_wgsim(fasta_file, os.path.join(output_dir, "reads", f"original_simulated_sequence.length={seq_length}"), read_length=150, num_reads=num_reads, error_rate=0.01)
    else:
        simulated_seq = str(next(SeqIO.parse(os.path.join(output_dir, "references", f"original_simulated_sequence.length={seq_length}.fasta"), "fasta")).seq)
    # step 2: create mutated versions of the sequence with 1, 2, 4, 8, 16, 32, 64, and 128 mutations
    for num_mutations in mutation_counts:
        mutated_fasta_file = os.path.join(output_dir, "references", f"simulated_sequence.length={seq_length}_with_{num_mutations}_mutations.fasta")
        if not os.path.exists(mutated_fasta_file):
            mutated_seq = mutate_sequence(simulated_seq, num_mutations)
            with open(mutated_fasta_file, "w") as f:
                f.write(f">simulated_sequence.length={seq_length}_with_{num_mutations}_mutations\n{mutated_seq}\n")
            # simulate reads from the mutated sequence
            simulate_reads_with_wgsim(mutated_fasta_file, os.path.join(output_dir, "reads", f"simulated_sequence.length={seq_length}_with_{num_mutations}_mutations"), read_length=150, num_reads=num_reads, error_rate=0.01)

def create_indexes_if_do_not_exist(kallisto_index_prefix: str, kma_index_prefix: str, fasta_file: str) -> None:
    if not os.path.exists(kallisto_index_prefix):
        build_kallisto_index(fasta_file, kallisto_index_prefix)
    if not os.path.exists(kma_index_prefix + ".comp.bf"): # KMA creates multiple files with different extensions, so just check for one of them
        build_kma_index(fasta_file, kma_index_prefix)

def return_kallisto_total_assigned_read_count(kallisto_df: pl.DataFrame) -> float:
    kallisto_clean_df = kallisto_df.with_columns(
        pl.col("kallisto_estimated_counts").cast(pl.Float64, strict=False).fill_null(0.0).alias("kallisto_estimated_counts")
    )
    return float(kallisto_clean_df["kallisto_estimated_counts"].sum())

def return_kma_total_assigned_read_count(kma_df: pl.DataFrame) -> float:
    # Safely coerce non-numeric values to null, then fill with 0.
    kma_clean_df = kma_df.with_columns(
        pl.col("kma_read_count").cast(pl.Float64, strict=False).fill_null(0.0).alias("kma_read_count")
    )
    return float(kma_clean_df["kma_read_count"].sum())/2 # divide by 2 because KMA counts reads, and we want fragments

def compute_mash_distance(fasta1: str, fasta2: str, kmer_size=31) -> float:
    seq1 = str(next(SeqIO.parse(fasta1, "fasta")).seq)
    seq2 = str(next(SeqIO.parse(fasta2, "fasta")).seq)
    kmers1 = set(seq1[i:i+kmer_size] for i in range(len(seq1) - kmer_size + 1))
    kmers2 = set(seq2[i:i+kmer_size] for i in range(len(seq2) - kmer_size + 1))
    union = kmers1 | kmers2
    intersection = kmers1 & kmers2
    jaccard_index = len(intersection) / len(union) if len(union) > 0 else 0.0
    if jaccard_index <= 0:
        return float(1.0)  # Maximum distance if no shared kmers
    elif jaccard_index >= 1:
        return float(0.0)  # Minimum distance if all kmers are shared
    else:
        mash_distance = -1.0 / kmer_size * np.log(2 * jaccard_index / (1 + jaccard_index))
        return mash_distance

def run_kmer_simulation_RA_D(output_dir: str, seq_length=1000, num_reads=1000, mutation_counts=list(range(1, 51, 1))):
    # first make an output directory for the results
    os.makedirs(os.path.join(output_dir, "indexes"), exist_ok=True)
    analysis_index_dir = os.path.join(output_dir, "indexes", "RA_D")
    os.makedirs(analysis_index_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "RA_D_results"), exist_ok=True)
    # step 1: simulate original and mutated references and reads
    generate_references_and_reads_for_simulation(output_dir, seq_length, num_reads, mutation_counts)
    fasta_file = os.path.join(output_dir, "references", f"original_simulated_sequence.length={seq_length}.fasta")
    # for every pair of original/mutated sequence, join fastas, build indexes, mix reads in equal proportions, and run quantification with both tools
    results = []
    for refs_flipped in [False, True]: # to check if order of references in the fasta file affects results
        for num_mutations in mutation_counts:
            mutated_fasta_file = os.path.join(output_dir, "references", f"simulated_sequence.length={seq_length}_with_{num_mutations}_mutations.fasta")
            combined_fasta_file = os.path.join(output_dir, "references", f"combined_original_length={seq_length}_and_{num_mutations}_mutations.fasta")
            if refs_flipped:
                # Swap the order of the references in the combined file
                with open(combined_fasta_file, "w") as f:
                    for record in SeqIO.parse(mutated_fasta_file, "fasta"):
                        SeqIO.write(record, f, "fasta")
                    for record in SeqIO.parse(fasta_file, "fasta"):
                        SeqIO.write(record, f, "fasta")
            else:
                with open(combined_fasta_file, "w") as f:
                    for record in SeqIO.parse(fasta_file, "fasta"):
                        SeqIO.write(record, f, "fasta")
                    for record in SeqIO.parse(mutated_fasta_file, "fasta"):
                        SeqIO.write(record, f, "fasta")
            # build indexes
            kallisto_index_prefix = os.path.join(analysis_index_dir, f"kallisto_index_original_length={seq_length}_and_with_{num_mutations}_mutations")
            kma_index_prefix = os.path.join(analysis_index_dir, f"kma_index_original_length={seq_length}_and_with_{num_mutations}_mutations")
            create_indexes_if_do_not_exist(kallisto_index_prefix, kma_index_prefix, combined_fasta_file)
            # mix reads in equal proportions
            original_read1 = os.path.join(output_dir, "reads", f"original_simulated_sequence.length={seq_length}_R1.fastq")
            original_read2 = os.path.join(output_dir, "reads", f"original_simulated_sequence.length={seq_length}_R2.fastq")
            mutated_read1 = os.path.join(output_dir, "reads", f"simulated_sequence.length={seq_length}_with_{num_mutations}_mutations_R1.fastq")
            mutated_read2 = os.path.join(output_dir, "reads", f"simulated_sequence.length={seq_length}_with_{num_mutations}_mutations_R2.fastq")
            mixed_read1 = os.path.join(output_dir, "reads", f"mixed_reads.length={seq_length}_with_{num_mutations}_mutations_R1.fastq")
            mixed_read2 = os.path.join(output_dir, "reads", f"mixed_reads.length={seq_length}_with_{num_mutations}_mutations_R2.fastq")
            mix_reads_in_equal_proportions(original_read1, original_read2, mutated_read1, mutated_read2, mixed_read1, mixed_read2)
            # run quantification with both tools
            kallisto_output_dir = os.path.join(output_dir, "RA_D_results", f"kallisto_output_{num_mutations}_mutations")
            kma_output_prefix = os.path.join(output_dir, "RA_D_results", f"kma_output_{num_mutations}_mutations")
            run_kallisto_quantification(kallisto_index_prefix, mixed_read1, mixed_read2, kallisto_output_dir)
            run_kma_quantification(kma_index_prefix, mixed_read1, mixed_read2, kma_output_prefix)
            # parse outputs
            kallisto_df = parse_kallisto_output(os.path.join(kallisto_output_dir, "abundance.tsv"))
            kma_df = parse_kma_output(f"{kma_output_prefix}.mapstat")
            # compute kmer metrics
            n_shared_kmers, n_total_kmers, n_kmers_unique_to_1, n_kmers_unique_to_2 = return_kmer_metrics(fasta_file, mutated_fasta_file, kmer_size=31)
            # extract read counts for original and mutated reference from both tools
            kallisto_count_original = return_kallisto_read_count_for_reference(kallisto_df, f"simulated_sequence.length={seq_length}")
            kallisto_count_mutated = return_kallisto_read_count_for_reference(kallisto_df, f"simulated_sequence.length={seq_length}_with_{num_mutations}_mutations")
            kma_count_original = return_kma_read_count_for_reference(kma_df, f"simulated_sequence.length={seq_length}")
            kma_count_mutated = return_kma_read_count_for_reference(kma_df, f"simulated_sequence.length={seq_length}_with_{num_mutations}_mutations")
            # compute log error between observed and expected ratios for both tools
            kallisto_log_error = compute_log_error(expected_A=2000, expected_B=2000, observed_A=kallisto_count_original, observed_B=kallisto_count_mutated)
            kma_log_error = compute_log_error(expected_A=2000, expected_B=2000, observed_A=kma_count_original, observed_B=kma_count_mutated)
            # compute distinguishibility metric D
            D = compute_distinguishibility_D(n_shared_kmers, n_kmers_unique_to_1, n_kmers_unique_to_2)
            # compute mash distance between original and mutated sequence
            mash_distance = compute_mash_distance(fasta_file, mutated_fasta_file, kmer_size=31)
            # add to results
            results.append({
                "num_mutations": num_mutations,
                "n_shared_kmers": n_shared_kmers,
                "n_total_kmers": n_total_kmers,
                "n_kmers_unique_to_original": n_kmers_unique_to_1,
                "n_kmers_unique_to_mutated": n_kmers_unique_to_2,
                "kallisto_count_original": kallisto_count_original,
                "kallisto_count_mutated": kallisto_count_mutated,
                "kallisto_log_error": kallisto_log_error,
                "kma_count_original": kma_count_original,
                "kma_count_mutated": kma_count_mutated,
                "kma_log_error": kma_log_error,
                "distinguishibility_D": D,
                "mash_distance": mash_distance,
                "ref_flipped": refs_flipped,
            })
    results_df = pl.DataFrame(results)
    file_name = f"kmer_simulation.RA_D.seq_length={seq_length}_num_reads={num_reads}.min_mutations={min(mutation_counts)}.max_mutations={max(mutation_counts)}.tsv"
    results_df.write_csv(os.path.join(output_dir, file_name),
                         separator="\t")
    
def run_kmer_simulation_ID_M(output_dir: str, seq_length=1000, num_reads=1000, mutation_counts=list(range(1, 51, 1))):
    # if do not exist, generate refs and reads
    generate_references_and_reads_for_simulation(output_dir, seq_length, num_reads, mutation_counts)
    # make a folder for the diversity results
    diversity_output_dir = os.path.join(output_dir, "ID_M_results")
    os.makedirs(diversity_output_dir, exist_ok=True)
    analysis_index_dir = os.path.join(output_dir, "indexes", "ID_M")
    os.makedirs(analysis_index_dir, exist_ok=True)
    # generate indexes for just the original sequence
    fasta_file = os.path.join(output_dir, "references", f"original_simulated_sequence.length={seq_length}.fasta")
    kallisto_index_prefix = os.path.join(analysis_index_dir, f"kallisto_index_original_length={seq_length}_only")
    kma_index_prefix = os.path.join(analysis_index_dir, f"kma_index_original_length={seq_length}_only")
    create_indexes_if_do_not_exist(kallisto_index_prefix, kma_index_prefix, fasta_file)  
    # for each mutated sequence, create a reference with the original only and run tools to see how many reads they assign to the original
    results = []
    for num_mutations in mutation_counts:
        # run quantification with both tools
        mutated_fasta_file = os.path.join(output_dir, "references", f"simulated_sequence.length={seq_length}_with_{num_mutations}_mutations.fasta")
        mutated_read1 = os.path.join(output_dir, "reads", f"simulated_sequence.length={seq_length}_with_{num_mutations}_mutations_R1.fastq")
        mutated_read2 = os.path.join(output_dir, "reads", f"simulated_sequence.length={seq_length}_with_{num_mutations}_mutations_R2.fastq")
        kallisto_output_dir = os.path.join(diversity_output_dir, f"kallisto_output_{num_mutations}_mutations")
        kma_output_prefix = os.path.join(diversity_output_dir, f"kma_output_{num_mutations}_mutations")
        run_kallisto_quantification(kallisto_index_prefix, mutated_read1, mutated_read2, kallisto_output_dir)
        run_kma_quantification(kma_index_prefix, mutated_read1, mutated_read2, kma_output_prefix)
        # parse outputs
        kallisto_df = parse_kallisto_output(os.path.join(kallisto_output_dir, "abundance.tsv"))
        kma_df = parse_kma_output(f"{kma_output_prefix}.mapstat")
        # extract read counts for original reference from both tools
        kallisto_count_original = return_kallisto_read_count_for_reference(kallisto_df, f"simulated_sequence.length={seq_length}")
        kma_count_original = return_kma_read_count_for_reference(kma_df, f"simulated_sequence.length={seq_length}")
        # compute kmer metrics
        n_shared_kmers, n_total_kmers, n_kmers_unique_to_1, n_kmers_unique_to_2 = return_kmer_metrics(fasta_file, mutated_fasta_file, kmer_size=31)
        # compute matching metric M
        M = compute_matching_M(fasta_file, mutated_fasta_file, kmer_size=31) # want proportion of kmers in mutated that are in original
        # compute proportions assinged for each tool
        kallisto_proportion_assigned = kallisto_count_original / num_reads
        kma_proportion_assigned = kma_count_original / num_reads
        # compute mash distance between original and mutated sequence
        mash_distance = compute_mash_distance(fasta_file, mutated_fasta_file, kmer_size=31)
        # add to results
        results.append({
            "num_mutations": num_mutations,
            "n_shared_kmers": n_shared_kmers,
            "n_total_kmers": n_total_kmers,
            "n_kmers_unique_to_original": n_kmers_unique_to_1,
            "n_kmers_unique_to_mutated": n_kmers_unique_to_2,
            "matching_M": M,
            "kallisto_count_original": kallisto_count_original,
            "kallisto_proportion_assigned": kallisto_proportion_assigned,
            "kma_count_original": kma_count_original,
            "kma_proportion_assigned": kma_proportion_assigned,
            "mash_distance": mash_distance,
        })
    results_df = pl.DataFrame(results)
    output_file_name = f"kmer_simulation.ID_M.seq_length={seq_length}_num_reads={num_reads}.min_mutations={min(mutation_counts)}.max_mutations={max(mutation_counts)}.tsv"
    results_df.write_csv(os.path.join(output_dir, output_file_name),
                         separator="\t")

def run_kmer_simulation_ID_D(output_dir: str, seq_length=1000, num_reads=1000, mutation_counts=list(range(1, 51, 1))):
    # if do not exist, generate refs and reads
    generate_references_and_reads_for_simulation(output_dir, seq_length, num_reads, mutation_counts)
    # make a folder for the diversity results
    diversity_output_dir = os.path.join(output_dir, "ID_D_results")
    os.makedirs(diversity_output_dir, exist_ok=True)
    analysis_index_dir = os.path.join(output_dir, "indexes", "ID_D")
    os.makedirs(analysis_index_dir, exist_ok=True)
    fasta_file = os.path.join(output_dir, "references", f"original_simulated_sequence.length={seq_length}.fasta")
    # for each mutated sequence, create a ref for mutated and original, then take original reads and see how many are assigned to original vs mutated, and how that relates to the distinguishibility D metric
    results = []
    for refs_flipped in [False, True]: # put mutated first in the fasta to check if order affects results
        ref_order_tag = "flipped" if refs_flipped else "normal"
        for num_mutations in mutation_counts:
            # make indexes for original + mutated reference
            mutated_fasta_file = os.path.join(output_dir, "references", f"simulated_sequence.length={seq_length}_with_{num_mutations}_mutations.fasta")
            combined_fasta_file = os.path.join(output_dir, "references", f"combined_original_length={seq_length}_and_{num_mutations}_mutations.{ref_order_tag}.fasta")
            if refs_flipped:
                # Swap the order of the references in the combined file
                with open(combined_fasta_file, "w") as f:
                    for record in SeqIO.parse(mutated_fasta_file, "fasta"):
                        SeqIO.write(record, f, "fasta")
                    for record in SeqIO.parse(fasta_file, "fasta"):
                        SeqIO.write(record, f, "fasta")
            else:
                with open(combined_fasta_file, "w") as f:
                    for record in SeqIO.parse(fasta_file, "fasta"):
                        SeqIO.write(record, f, "fasta")
                    for record in SeqIO.parse(mutated_fasta_file, "fasta"):
                        SeqIO.write(record, f, "fasta")
            kallisto_index_prefix = os.path.join(analysis_index_dir, f"kallisto_index_original_length={seq_length}_and_with_{num_mutations}_mutations.{ref_order_tag}")
            kma_index_prefix = os.path.join(analysis_index_dir, f"kma_index_original_length={seq_length}_and_with_{num_mutations}_mutations.{ref_order_tag}")
            create_indexes_if_do_not_exist(kallisto_index_prefix, kma_index_prefix, combined_fasta_file)
            # run quantification on original reads with both tools
            original_read1 = os.path.join(output_dir, "reads", f"original_simulated_sequence.length={seq_length}_R1.fastq")
            original_read2 = os.path.join(output_dir, "reads", f"original_simulated_sequence.length={seq_length}_R2.fastq")
            kallisto_output_dir = os.path.join(diversity_output_dir, f"kallisto_output_{num_mutations}_mutations.{ref_order_tag}")
            kma_output_prefix = os.path.join(diversity_output_dir, f"kma_output_{num_mutations}_mutations.{ref_order_tag}")
            run_kallisto_quantification(kallisto_index_prefix, original_read1, original_read2, kallisto_output_dir)
            run_kma_quantification(kma_index_prefix, original_read1, original_read2, kma_output_prefix)
            # parse outputs
            kallisto_df = parse_kallisto_output(os.path.join(kallisto_output_dir, "abundance.tsv"))
            kma_df = parse_kma_output(f"{kma_output_prefix}.mapstat")
            # extract read counts for original reference from both tools
            kallisto_count_original = return_kallisto_read_count_for_reference(kallisto_df, f"simulated_sequence.length={seq_length}")
            kma_count_original = return_kma_read_count_for_reference(kma_df, f"simulated_sequence.length={seq_length}")
            kallisto_total_assigned = return_kallisto_total_assigned_read_count(kallisto_df)
            kma_total_assigned = return_kma_total_assigned_read_count(kma_df)
            # compute kmer metrics
            n_shared_kmers, n_total_kmers, n_kmers_unique_to_1, n_kmers_unique_to_2 = return_kmer_metrics(fasta_file, mutated_fasta_file, kmer_size=31)
            # compute distinguishibility metric D
            D = compute_distinguishibility_D(n_shared_kmers, n_kmers_unique_to_1, n_kmers_unique_to_2)
            # compute proportions assinged for each tool
            kallisto_proportion_assigned = kallisto_count_original / kallisto_total_assigned if kallisto_total_assigned > 0 else 0.0
            kma_proportion_assigned = kma_count_original / kma_total_assigned if kma_total_assigned > 0 else 0.0
            # compute mash distance between original and mutated sequence
            mash_distance = compute_mash_distance(fasta_file, mutated_fasta_file, kmer_size=31)
            # add to results
            results.append({
                "num_mutations": num_mutations,
                "n_shared_kmers": n_shared_kmers,
                "n_total_kmers": n_total_kmers,
                "n_kmers_unique_to_original": n_kmers_unique_to_1,
                "n_kmers_unique_to_mutated": n_kmers_unique_to_2,
                "distinguishibility_D": D,
                "kallisto_count_original": kallisto_count_original,
                "kallisto_proportion_assigned": kallisto_proportion_assigned,
                "kma_count_original": kma_count_original,
                "kma_proportion_assigned": kma_proportion_assigned,
                "refs_flipped": refs_flipped,
                "mash_distance": mash_distance,
            })
    results_df = pl.DataFrame(results)
    output_file_name = f"kmer_simulation.ID_D.seq_length={seq_length}_num_reads={num_reads}.min_mutations={min(mutation_counts)}.max_mutations={max(mutation_counts)}.tsv"
    results_df.write_csv(os.path.join(output_dir, output_file_name),
                         separator="\t")

def run_kmer_simulation_ID_DM(output_dir: str, seq_length=1000, num_reads=1000, mutation_counts=list(range(1, 51, 1))):
    # if do not exist, generate refs and reads
    generate_references_and_reads_for_simulation(output_dir, seq_length, num_reads, mutation_counts=mutation_counts)
    # make a folder for the diversity results
    diversity_output_dir = os.path.join(output_dir, "ID_DM_results")
    os.makedirs(diversity_output_dir, exist_ok=True)
    analysis_index_dir = os.path.join(output_dir, "indexes", "ID_DM")
    os.makedirs(analysis_index_dir, exist_ok=True)
    fasta_file = os.path.join(output_dir, "references", f"original_simulated_sequence.length={seq_length}.fasta")
    # this is the most complicated analysis. There is a tree with three branches: S (sample), then A and B (the references)
    # I am going to assess what happens when D between A and B changes, and M between S and B (closest target) changes.
    # A will be the original sequence, add some mutations to create B, then add mutations to create S. Generate reads for all.
    # We will runt tools and quant assignments on A and B, and compute D and M
    # final metric we want is proportion assigned to B over total assigned to A and B, and how that relates to D and M
    A_fasta_file = os.path.join(output_dir, "references", f"original_simulated_sequence.length={seq_length}.fasta")
    results = []
    # for multithreading to speed things up
    def process_inner_loop(num_mutations_i, num_mutations_j, B_fasta_file, B2_fasta_file, kallisto_index_prefix, kma_index_prefix):
        """Process a single inner loop iteration for a given mutation pair."""
        # add more mutations to B to create S
        S_fasta_file = os.path.join(output_dir, "references", f"simulated_sequence_with_{num_mutations_i}.length={seq_length}_with_{num_mutations_j}_mutations.fasta")
        S_sequence = mutate_sequence(str(next(SeqIO.parse(B_fasta_file, "fasta")).seq), num_mutations_j)
        with open(S_fasta_file, "w") as f:
            f.write(f">simulated_sequence_with_{num_mutations_i}.length={seq_length}_with_{num_mutations_j}_mutations\n{S_sequence}\n")
        # generate reads for S
        simulate_reads_with_wgsim(S_fasta_file, os.path.join(output_dir, "reads", f"simulated_sequence_with_{num_mutations_i}.length={seq_length}_with_{num_mutations_j}_mutations"), read_length=150, num_reads=num_reads, error_rate=0.01)
        # run quantification on S reads with both tools
        S_read1 = os.path.join(output_dir, "reads", f"simulated_sequence_with_{num_mutations_i}.length={seq_length}_with_{num_mutations_j}_mutations_R1.fastq")
        S_read2 = os.path.join(output_dir, "reads", f"simulated_sequence_with_{num_mutations_i}.length={seq_length}_with_{num_mutations_j}_mutations_R2.fastq")
        kallisto_output_dir = os.path.join(diversity_output_dir, f"kallisto_output_S_with_{num_mutations_i}_{num_mutations_j}_mutations")
        kma_output_prefix = os.path.join(diversity_output_dir, f"kma_output_S_with_{num_mutations_i}_{num_mutations_j}_mutations")
        run_kallisto_quantification(kallisto_index_prefix, S_read1, S_read2, kallisto_output_dir)
        run_kma_quantification(kma_index_prefix, S_read1, S_read2, kma_output_prefix)
        kallisto_output_file = os.path.join(kallisto_output_dir, "abundance.tsv")
        kma_output_file = f"{kma_output_prefix}.mapstat"
        if not os.path.exists(kallisto_output_file):
            raise FileNotFoundError(f"Missing kallisto output file: {kallisto_output_file}")
        if not os.path.exists(kma_output_file):
            raise FileNotFoundError(f"Missing KMA output file: {kma_output_file}")
        # parse outputs
        kallisto_df = parse_kallisto_output(kallisto_output_file)
        kma_df = parse_kma_output(kma_output_file)
        # extract read counts for A and B2 reference from both tools
        b2_reference = f"simulated_sequence_with_{num_mutations_i}_B2.length={seq_length}_with_{int(0.01 * seq_length)}_mutations"
        kallisto_count_A = return_kallisto_read_count_for_reference(kallisto_df, f"simulated_sequence.length={seq_length}")
        kallisto_count_B = return_kallisto_read_count_for_reference(kallisto_df, b2_reference)
        kma_count_A = return_kma_read_count_for_reference(kma_df, f"simulated_sequence.length={seq_length}")
        kma_count_B = return_kma_read_count_for_reference(kma_df, b2_reference)
        kallisto_total_assigned = return_kallisto_total_assigned_read_count(kallisto_df)
        kma_total_assigned = return_kma_total_assigned_read_count(kma_df)
        # compute kmer metrics between S and B2 (sister-target scenario)
        n_shared_kmers_SB2, n_total_kmers_SB2, n_kmers_unique_to_S, n_kmers_unique_to_B2 = return_kmer_metrics(S_fasta_file, B2_fasta_file, kmer_size=31)
        # compute kmer metrics between A and B2 
        n_shared_kmers_AB2, n_total_kmers_AB2, n_kmers_unique_to_A, n_kmers_unique_to_B2 = return_kmer_metrics(A_fasta_file, B2_fasta_file, kmer_size=31)
        # compute for A to S, so can ensure that S is closer to B2 than A is, which is important for the interpretation of D and M
        n_shared_kmers_AS, n_total_kmers_AS, n_kmers_unique_to_A, n_kmers_unique_to_S = return_kmer_metrics(A_fasta_file, S_fasta_file, kmer_size=31)
        # compute distinguishibility metric D between A and B
        D_AB = compute_distinguishibility_D(n_shared_kmers_AB2, n_kmers_unique_to_A, n_kmers_unique_to_B2)
        M_AB = compute_matching_M(A_fasta_file, B2_fasta_file, kmer_size=31)
        # compute matching metric M between S and B2
        M_SB = compute_matching_M(S_fasta_file, B2_fasta_file, kmer_size=31)
        D_SB = compute_distinguishibility_D(n_shared_kmers_SB2, n_kmers_unique_to_S, n_kmers_unique_to_B2)
        # both for A and S
        D_AS = compute_distinguishibility_D(n_shared_kmers_AS, n_kmers_unique_to_A, n_kmers_unique_to_S)
        M_AS = compute_matching_M(A_fasta_file, S_fasta_file, kmer_size=31)
        # compute proportions assigned to B over total assigned to A and B for both tools
        kallisto_proportion_assigned_to_B = kallisto_count_B / kallisto_total_assigned if kallisto_total_assigned > 0 else 0.0
        kma_proportion_assigned_to_B = kma_count_B / kma_total_assigned if kma_total_assigned > 0 else 0.0
        # also compute mash distances
        mash_AB2 = compute_mash_distance(A_fasta_file, B2_fasta_file, kmer_size=31)
        mash_SB2 = compute_mash_distance(S_fasta_file, B2_fasta_file, kmer_size=31)
        mash_AS = compute_mash_distance(A_fasta_file, S_fasta_file, kmer_size=31)
        # return result dict
        return {
            "num_mutations_in_B": num_mutations_i,
            "num_mutations_in_S": num_mutations_j,
            "n_shared_kmers_SB2": n_shared_kmers_SB2,
            "n_total_kmers_SB2": n_total_kmers_SB2,
            "n_kmers_unique_to_S": n_kmers_unique_to_S,
            "n_kmers_unique_to_B2": n_kmers_unique_to_B2,
            "n_shared_kmers_AB2": n_shared_kmers_AB2,
            "n_total_kmers_AB2": n_total_kmers_AB2,
            "n_kmers_unique_to_A": n_kmers_unique_to_A,
            "n_kmers_unique_to_B2": n_kmers_unique_to_B2,
            "distinguishibility_D_AB": D_AB,
            "matching_M_SB": M_SB,
            "distinguishibility_D_SB": D_SB,
            "matching_M_AS": M_AS,
            "distinguishibility_D_AS": D_AS,
            "kallisto_count_A": kallisto_count_A,
            "kallisto_count_B": kallisto_count_B,
            "kallisto_proportion_assigned_to_B": kallisto_proportion_assigned_to_B,
            "kma_count_A": kma_count_A,
            "kma_count_B": kma_count_B,
            "kma_proportion_assigned_to_B": kma_proportion_assigned_to_B,
            "kallisto_proportion_assigned" : kallisto_total_assigned / num_reads,
            "kma_proportion_assigned" : kma_total_assigned / num_reads,
            "mash_AB": mash_AB2,
            "mash_SB": mash_SB2,
            "mash_AS": mash_AS,
        }
    for num_mutations_i in mutation_counts:
        B_fasta_file = os.path.join(output_dir, "references", f"simulated_sequence.length={seq_length}_with_{num_mutations_i}_mutations.fasta")
        # build B2 once per i so all inner-loop jobs share the same sibling reference
        n_mutations_to_add_to_B = int(0.01 * seq_length)
        B2_fasta_file = os.path.join(output_dir, "references", f"simulated_sequence_with_{num_mutations_i}_B2.length={seq_length}_with_{n_mutations_to_add_to_B}_mutations.fasta")
        B_sequence = mutate_sequence(str(next(SeqIO.parse(B_fasta_file, "fasta")).seq), n_mutations_to_add_to_B)
        with open(B2_fasta_file, "w") as f:
            f.write(f">simulated_sequence_with_{num_mutations_i}_B2.length={seq_length}_with_{n_mutations_to_add_to_B}_mutations\n{B_sequence}\n")
        # build combined reference with A and B2 only
        combined_fasta_file = os.path.join(output_dir, "references", f"combined_A_B2_original_length={seq_length}_and_with_{num_mutations_i}_mutations.fasta")
        with open(combined_fasta_file, "w") as f:
            for record in SeqIO.parse(A_fasta_file, "fasta"):
                SeqIO.write(record, f, "fasta")
            for record in SeqIO.parse(B2_fasta_file, "fasta"):
                SeqIO.write(record, f, "fasta")
        # build indexes
        kallisto_index_prefix = os.path.join(analysis_index_dir, f"kallisto_index_A_B2_original_length={seq_length}_and_with_{num_mutations_i}_mutations")
        kma_index_prefix = os.path.join(analysis_index_dir, f"kma_index_A_B2_original_length={seq_length}_and_with_{num_mutations_i}_mutations")
        create_indexes_if_do_not_exist(kallisto_index_prefix, kma_index_prefix, combined_fasta_file)
        # Parallelize inner loop with threading
        max_workers = max(1, min(8, (os.cpu_count() or 8) // 2))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    process_inner_loop,
                    num_mutations_i,
                    j,
                    B_fasta_file,
                    B2_fasta_file,
                    kallisto_index_prefix,
                    kma_index_prefix,
                )
                for j in mutation_counts
            ]
            for future in as_completed(futures):
                results.append(future.result())
    results_df = pl.DataFrame(results)
    output_file_name = f"kmer_simulation.ID_DM.seq_length={seq_length}_num_reads={num_reads}.min_mutations={min(mutation_counts)}.max_mutations={max(mutation_counts)}.tsv"
    results_df.write_csv(os.path.join(output_dir, output_file_name),
                         separator="\t")

def run_kmer_simulation_RA_prop(output_dir: str, seq_length=1000, num_reads=1000, mutation_count=25):
    # if do not exist, generate refs and reads
    generate_references_and_reads_for_simulation(output_dir, seq_length, num_reads, mutation_counts=[mutation_count])
    # make a folder for the diversity results
    diversity_output_dir = os.path.join(output_dir, "RA_prop_results")
    os.makedirs(diversity_output_dir, exist_ok=True)
    analysis_index_dir = os.path.join(output_dir, "indexes", "RA_prop")
    os.makedirs(analysis_index_dir, exist_ok=True)
    fasta_file = os.path.join(output_dir, "references", f"original_simulated_sequence.length={seq_length}.fasta")
    # instead of iterating over mutated sequences, going to fix number of mutations and change proportions of reads from original vs mutated, and see how that changes error
    mutated_fasta_file = os.path.join(output_dir, "references", f"simulated_sequence.length={seq_length}_with_{mutation_count}_mutations.fasta")
    # build combined reference
    combined_fasta_file = os.path.join(output_dir, "references", f"combined_original_length={seq_length}_and_{mutation_count}_mutations.fasta")
    with open(combined_fasta_file, "w") as f:
        for record in SeqIO.parse(fasta_file, "fasta"):
            SeqIO.write(record, f, "fasta")
        for record in SeqIO.parse(mutated_fasta_file, "fasta"):
            SeqIO.write(record, f, "fasta")
    kallisto_index_prefix = os.path.join(analysis_index_dir, f"kallisto_index_original_length={seq_length}_and_with_{mutation_count}_mutations")
    kma_index_prefix = os.path.join(analysis_index_dir, f"kma_index_original_length={seq_length}_and_with_{mutation_count}_mutations")
    create_indexes_if_do_not_exist(kallisto_index_prefix, kma_index_prefix, combined_fasta_file)
    # Build one high-depth read pool once, then subsample for each proportion.
    min_proportion = 0.01
    min_number_of_reads_for_5fold = int(np.ceil((5 * seq_length) / (150 * 2)))
    total_read_pairs_to_simulate = int(np.ceil(min_number_of_reads_for_5fold / min_proportion))
    total_read_pairs_to_simulate = max(total_read_pairs_to_simulate, num_reads)
    os.makedirs(os.path.join(output_dir, "reads", "extra_reads"), exist_ok=True)
    simulate_reads_with_wgsim(
        fasta_file,
        os.path.join(output_dir, "reads", "extra_reads", f"original_simulated_sequence.length={seq_length}_extra"),
        read_length=150,
        num_reads=total_read_pairs_to_simulate,
        error_rate=0.01,
    )
    simulate_reads_with_wgsim(
        mutated_fasta_file,
        os.path.join(output_dir, "reads", "extra_reads", f"simulated_sequence.length={seq_length}_with_{mutation_count}_mutations_extra"),
        read_length=150,
        num_reads=total_read_pairs_to_simulate,
        error_rate=0.01,
    )
    results = []
    for proportion_original in np.arange(0.01, 1.0, 0.01):
        proportion_mutated = 1 - proportion_original
        # mix reads in specified proportions
        original_read1 = os.path.join(output_dir, "reads", "extra_reads", f"original_simulated_sequence.length={seq_length}_extra_R1.fastq")
        original_read2 = os.path.join(output_dir, "reads", "extra_reads", f"original_simulated_sequence.length={seq_length}_extra_R2.fastq")
        mutated_read1 = os.path.join(output_dir, "reads", "extra_reads", f"simulated_sequence.length={seq_length}_with_{mutation_count}_mutations_extra_R1.fastq")
        mutated_read2 = os.path.join(output_dir, "reads", "extra_reads", f"simulated_sequence.length={seq_length}_with_{mutation_count}_mutations_extra_R2.fastq")
        mixed_read1 = os.path.join(output_dir, "reads", f"mixed_reads.prop_original={proportion_original:.2f}_and_prop_mutated={proportion_mutated:.2f}_R1.fastq")
        mixed_read2 = os.path.join(output_dir, "reads", f"mixed_reads.prop_original={proportion_original:.2f}_and_prop_mutated={proportion_mutated:.2f}_R2.fastq")
        # to mix in specified proportions, can use head and tail to take specified number of reads from each file
        num_reads_original = int(proportion_original * total_read_pairs_to_simulate)
        num_reads_mutated = int(proportion_mutated * total_read_pairs_to_simulate)
        command_mix_read1 = f"(head -n {num_reads_original*4} {original_read1} && head -n {num_reads_mutated*4} {mutated_read1}) > {mixed_read1}"
        command_mix_read2 = f"(head -n {num_reads_original*4} {original_read2} && head -n {num_reads_mutated*4} {mutated_read2}) > {mixed_read2}"
        subprocess.run(command_mix_read1, shell=True, check=True)
        subprocess.run(command_mix_read2, shell=True, check=True)
        # run quantification with both tools
        kallisto_output_dir = os.path.join(diversity_output_dir, f"kallisto_output_prop_original={proportion_original:.2f}")
        kma_output_prefix = os.path.join(diversity_output_dir, f"kma_output_prop_original={proportion_original:.2f}")
        run_kallisto_quantification(kallisto_index_prefix, mixed_read1, mixed_read2, kallisto_output_dir)
        run_kma_quantification(kma_index_prefix, mixed_read1, mixed_read2, kma_output_prefix)
        # parse outputs
        kallisto_df = parse_kallisto_output(os.path.join(kallisto_output_dir, "abundance.tsv"))
        kma_df = parse_kma_output(f"{kma_output_prefix}.mapstat")
        # extract read counts for original and mutated reference from both tools
        kallisto_count_original = return_kallisto_read_count_for_reference(kallisto_df, f"simulated_sequence.length={seq_length}")
        kallisto_count_mutated = return_kallisto_read_count_for_reference(kallisto_df, f"simulated_sequence.length={seq_length}_with_{mutation_count}_mutations")
        kma_count_original = return_kma_read_count_for_reference(kma_df, f"simulated_sequence.length={seq_length}")
        kma_count_mutated = return_kma_read_count_for_reference(kma_df, f"simulated_sequence.length={seq_length}_with_{mutation_count}_mutations")
        kallisto_total_assigned = return_kallisto_total_assigned_read_count(kallisto_df)
        kma_total_assigned = return_kma_total_assigned_read_count(kma_df)
        # compute log error between observed and expected ratios for both tools
        kallisto_expected_count_original = proportion_original * kallisto_total_assigned * 2 # times 2 because we are simulating paired end reads, so total number of reads is num_reads * 2
        kallisto_expected_count_mutated = proportion_mutated * kallisto_total_assigned * 2
        kma_expected_count_original = proportion_original * kma_total_assigned * 2
        kma_expected_count_mutated = proportion_mutated * kma_total_assigned * 2
        kallisto_log_error = compute_log_error(expected_A=kallisto_expected_count_original, expected_B=kallisto_expected_count_mutated, observed_A=kallisto_count_original, observed_B=kallisto_count_mutated)
        kma_log_error = compute_log_error(expected_A=kma_expected_count_original, expected_B=kma_expected_count_mutated, observed_A=kma_count_original, observed_B=kma_count_mutated)
        # add to results
        results.append({
            "proportion_original": proportion_original,
            "proportion_mutated": proportion_mutated,
            "kallisto_count_original": kallisto_count_original,
            "kallisto_count_mutated": kallisto_count_mutated,
            "kallisto_log_error": kallisto_log_error,
            "kma_count_original": kma_count_original,
            "kma_count_mutated": kma_count_mutated,
            "kma_log_error": kma_log_error,
        })
    results_df = pl.DataFrame(results)
    output_file_name = f"kmer_simulation.RA_prop.seq_length={seq_length}_num_reads={num_reads}_with_{mutation_count}_mutations.tsv"
    results_df.write_csv(os.path.join(output_dir, output_file_name),
                         separator="\t")

def run_kmer_simulation_RA_deltaM(output_dir: str, seq_length=1000, num_reads=1000, mutation_counts=list(range(1, 51, 1))):
    # if do not exist, generate refs and reads
    generate_references_and_reads_for_simulation(output_dir, seq_length, num_reads, mutation_counts=mutation_counts)
    # make a folder for the diversity results
    diversity_output_dir = os.path.join(output_dir, "RA_deltaM_results")
    os.makedirs(diversity_output_dir, exist_ok=True)
    analysis_index_dir = os.path.join(output_dir, "indexes", "RA_deltaM")
    os.makedirs(analysis_index_dir, exist_ok=True)
    first_fasta_file = os.path.join(output_dir, "references", f"original_simulated_sequence.length={seq_length}.fasta")
    # simulate a second sequence and mutate
    os.makedirs(os.path.join(output_dir, "second_references"), exist_ok=True)
    generate_references_and_reads_for_simulation(os.path.join(output_dir, "second_references"), seq_length, num_reads, mutation_counts=mutation_counts)
    # make indexs for first original and second original sequence
    second_fasta_file = os.path.join(output_dir, "second_references", "references", f"original_simulated_sequence.length={seq_length}.fasta")
    combined_fasta_file = os.path.join(output_dir, "references", f"combined_first_original_and_second_original_length={seq_length}.fasta")
    with open(combined_fasta_file, "w") as f:
        for record in SeqIO.parse(first_fasta_file, "fasta"):
            SeqIO.write(record, f, "fasta")
        for record in SeqIO.parse(second_fasta_file, "fasta"):
            # change id so is different
            record.id = record.id + "_second"
            record.description = record.id
            SeqIO.write(record, f, "fasta")
    kallisto_index_prefix = os.path.join(analysis_index_dir, f"kallisto_index_first_original_and_second_original_length={seq_length}")
    kma_index_prefix = os.path.join(analysis_index_dir, f"kma_index_first_original_and_second_original_length={seq_length}")
    create_indexes_if_do_not_exist(kallisto_index_prefix, kma_index_prefix, combined_fasta_file)
    # for first then second original sequence, take original reads plus mutated of the other, and quant
    results = []
    for original_seq in [first_fasta_file, second_fasta_file]:
        if original_seq == first_fasta_file:
            reads_dir = os.path.join(output_dir)
            other_reads_dir = os.path.join(output_dir, "second_references", "reads")
            other_fasta_file = second_fasta_file
            name = "first_original"
        else:
            reads_dir = os.path.join(output_dir, "second_references")
            other_reads_dir = os.path.join(output_dir, "reads")
            other_fasta_file = first_fasta_file
            name = "second_original"
        for num_mutations in mutation_counts:
            mutated_fasta_file = os.path.join(reads_dir, "references", f"simulated_sequence.length={seq_length}_with_{num_mutations}_mutations.fasta")
            mutated_read1 = os.path.join(reads_dir, "reads", f"simulated_sequence.length={seq_length}_with_{num_mutations}_mutations_R1.fastq")
            mutated_read2 = os.path.join(reads_dir, "reads", f"simulated_sequence.length={seq_length}_with_{num_mutations}_mutations_R2.fastq")
            original_read1 = os.path.join(other_reads_dir, f"original_simulated_sequence.length={seq_length}_R1.fastq")
            original_read2 = os.path.join(other_reads_dir, f"original_simulated_sequence.length={seq_length}_R2.fastq")
            mixed_read1 = os.path.join(reads_dir, "reads", f"mixed_reads.{name}.deltaM_prop_original=0.5_and_prop_mutated=0.5_R1.fastq")
            mixed_read2 = os.path.join(reads_dir, "reads", f"mixed_reads.{name}.deltaM_prop_original=0.5_and_prop_mutated=0.5_R2.fastq")
            if not os.path.exists(original_read1) or not os.path.exists(original_read2):
                raise FileNotFoundError(f"Missing original reads for mixing: {original_read1}, {original_read2}")
            if not os.path.exists(mutated_read1) or not os.path.exists(mutated_read2):
                raise FileNotFoundError(f"Missing mutated reads for mixing: {mutated_read1}, {mutated_read2}")
            mix_reads_in_equal_proportions(original_read1, original_read2, mutated_read1, mutated_read2, mixed_read1, mixed_read2)
            # run quantification with both tools
            kallisto_output_dir = os.path.join(diversity_output_dir, f"kallisto_output_{name}_with_{num_mutations}_mutations")
            kma_output_prefix = os.path.join(diversity_output_dir, f"kma_output_{name}_with_{num_mutations}_mutations")
            run_kallisto_quantification(kallisto_index_prefix, mixed_read1, mixed_read2, kallisto_output_dir)
            run_kma_quantification(kma_index_prefix, mixed_read1, mixed_read2, kma_output_prefix)
            # parse outputs
            kallisto_df = parse_kallisto_output(os.path.join(kallisto_output_dir, "abundance.tsv"))
            kma_df = parse_kma_output(f"{kma_output_prefix}.mapstat")
            # extract read counts for original and mutated reference from both tools
            kallisto_count_first_original = return_kallisto_read_count_for_reference(kallisto_df, f"simulated_sequence.length={seq_length}")
            kallisto_count_second_original = return_kallisto_read_count_for_reference(kallisto_df, f"simulated_sequence.length={seq_length}_second")
            kma_count_first_original = return_kma_read_count_for_reference(kma_df, f"simulated_sequence.length={seq_length}")
            kma_count_second_original = return_kma_read_count_for_reference(kma_df, f"simulated_sequence.length={seq_length}_second")
            kallisto_total_assigned = return_kallisto_total_assigned_read_count(kallisto_df)
            kma_total_assigned = return_kma_total_assigned_read_count(kma_df)
            # compute log error between observed and expected ratios for both tools
            kallisto_expected_count_first_original = 0.5 * kallisto_total_assigned * 2 # times 2 because we are simulating paired end reads, so total number of reads is num_reads * 2
            kallisto_expected_count_second_original = 0.5 * kallisto_total_assigned * 2
            kma_expected_count_first_original = 0.5 * kma_total_assigned * 2
            kma_expected_count_second_original = 0.5 * kma_total_assigned * 2
            kallisto_log_error = compute_log_error(expected_A=kallisto_expected_count_first_original, expected_B=kallisto_expected_count_second_original, observed_A=kallisto_count_first_original, observed_B=kallisto_count_second_original)
            kma_log_error = compute_log_error(expected_A=kma_expected_count_first_original, expected_B=kma_expected_count_second_original, observed_A=kma_count_first_original, observed_B=kma_count_second_original)
            # compute matching metric M between mutated sequence and original sequence
            M1 = compute_matching_M(mutated_fasta_file, original_seq, kmer_size=31)
            M2 = compute_matching_M(other_fasta_file, other_fasta_file, kmer_size=31) # should be 1
            delta_M = M2 - M1
            mash_original = compute_mash_distance(mutated_fasta_file, original_seq, kmer_size=31)
            mash_other = compute_mash_distance(other_fasta_file, other_fasta_file, kmer_size=31) # should be 0
            # add to results
            results.append({
                "mutated_sequence": name,
                "num_mutations": num_mutations,
                "M1": M1,
                "M2": M2,
                "delta_M": delta_M,
                "kallisto_count_first": kallisto_count_first_original,
                "kallisto_count_second": kallisto_count_second_original,
                "kallisto_log_error": kallisto_log_error,
                "kma_count_first": kma_count_first_original,
                "kma_count_second": kma_count_second_original,
                "kma_log_error": kma_log_error,
                "mash_where_differ": mash_original,
                "mash_where_match": mash_other
            })
    results_df = pl.DataFrame(results)
    output_file_name = f"kmer_simulation.RA_deltaM.seq_length={seq_length}_num_reads={num_reads}.min_mutations={min(mutation_counts)}.max_mutations={max(mutation_counts)}.tsv"
    results_df.write_csv(os.path.join(output_dir, output_file_name),
                         separator="\t")