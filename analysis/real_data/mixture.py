import subprocess
from .process import count_reads_in_fastq, load_json, make_dict_of_refs
from ..simulations.process import (
    build_kma_index,
    run_kma_quantification,
    build_kallisto_index,
    run_kallisto_quantification,
    parse_kma_output,
    parse_kallisto_output,
    compute_log_error
)
import polars as pl
import os
from ..shared.mosquito_ids import return_species_from_target_id

def make_all_unique_sample_pairs(samples: list) -> list:
    pairs = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            pairs.append((samples[i], samples[j]))
    return pairs

def get_number_of_reads_to_sample(R1_A: str, R1_B: str, proportion_A: float) -> tuple:
    total_reads_A = count_reads_in_fastq(R1_A)
    total_reads_B = count_reads_in_fastq(R1_B)
    reads_to_sample_A = int(proportion_A * min(total_reads_A, total_reads_B))
    reads_to_sample_B = int((1 - proportion_A) * min(total_reads_A, total_reads_B))
    return reads_to_sample_A, reads_to_sample_B

def mix_reads_in_proportion(R1_A: str, R2_A: str, R1_B: str, R2_B: str, output_R1: str, output_R2: str, proportion_A: float) -> tuple:
    reads_to_sample_A, reads_to_sample_B = get_number_of_reads_to_sample(R1_A, R1_B, proportion_A)
    # sample reads using seqtk
    command_R1 = f"seqtk sample -s100 {R1_A} {reads_to_sample_A} > {output_R1.replace('.gz', '')}"
    command_R2 = f"seqtk sample -s100 {R2_A} {reads_to_sample_A} > {output_R2.replace('.gz', '')}"
    subprocess.run(command_R1, shell=True, executable="/bin/bash", check=True)
    subprocess.run(command_R2, shell=True, executable="/bin/bash", check=True)
    command_R1_B = f"seqtk sample -s100 {R1_B} {reads_to_sample_B} >> {output_R1.replace('.gz', '')}"
    command_R2_B = f"seqtk sample -s100 {R2_B} {reads_to_sample_B} >> {output_R2.replace('.gz', '')}"
    subprocess.run(command_R1_B, shell=True, executable="/bin/bash", check=True)
    subprocess.run(command_R2_B, shell=True, executable="/bin/bash", check=True)
    # gzip the output files
    subprocess.run(f"gzip {output_R1.replace('.gz', '')}", shell=True, executable="/bin/bash", check=True)
    subprocess.run(f"gzip {output_R2.replace('.gz', '')}", shell=True, executable="/bin/bash", check=True)
    return reads_to_sample_A, reads_to_sample_B

def sum_across_species(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("reference").map_elements(lambda x: return_species_from_target_id(x).replace(" ", "_") if return_species_from_target_id(x) is not None else x).alias("reference")
    )
    if "kma_read_count" in df.columns:
        df = df.with_columns(
            pl.col("kma_read_count").sum().over("reference").alias("kma_read_count")
        )
    if "kallisto_estimated_counts" in df.columns:
        df = df.with_columns(
            pl.col("kallisto_estimated_counts").sum().over("reference").alias("kallisto_estimated_counts")
        )
    df = df.unique(subset="reference")
    return df

def make_mixtures_for_pairs_in_set_and_perform_abundance(abundance_fastas: list, sample_json: str, output_dir: str) -> pl.DataFrame:
    # build indexes for each fasta file
    ref_dict = make_dict_of_refs(abundance_fastas)
    for i, fasta_file in enumerate(abundance_fastas):
        if i == 0:
            index_type = "mosquito"
        else:
            index_type = "SARS-Cov2"
        index_prefix = f"{output_dir}/{index_type}_index"
        build_kma_index(fasta_file, index_prefix)
        build_kallisto_index(fasta_file, index_prefix)
    config = load_json(sample_json)
    results = []
    for dataset in config:
        kma_index = f"{output_dir}/{dataset}_index"
        kallisto_index = f"{output_dir}/{dataset}_index"
        samples = config[dataset]["samples"]
        pairs = make_all_unique_sample_pairs(list(samples.keys()))
        for sample_A, sample_B in pairs:
            R1_A = config[dataset]["samples"][sample_A]["R1"]
            R2_A = config[dataset]["samples"][sample_A]["R2"]
            R1_B = config[dataset]["samples"][sample_B]["R1"]
            R2_B = config[dataset]["samples"][sample_B]["R2"]
            for proportion_A in [0.1, 0.25, 0.5, 0.75, 0.9]:
                output_R1 = f"{output_dir}/mixture_{dataset}_{sample_A}_{sample_B}_propA_{proportion_A}_R1.fastq.gz"
                output_R2 = f"{output_dir}/mixture_{dataset}_{sample_A}_{sample_B}_propA_{proportion_A}_R2.fastq.gz"
                # run abundance estimation on the mixture (both kallisto and kma)
                output_for_sample = f"{output_dir}/{dataset}_{sample_A}_{sample_B}_propA_{proportion_A}"
                kma_output = f"{output_for_sample}.mapstat"
                kallisto_output = os.path.join(output_for_sample, "abundance.tsv")
                # run if not run yet
                if not (os.path.exists(kma_output) and os.path.exists(kallisto_output)):
                    expA, expB = mix_reads_in_proportion(R1_A, R2_A, R1_B, R2_B, output_R1, output_R2, proportion_A)
                    assert count_reads_in_fastq(output_R1) == expA + expB, f"Expected {expA + expB} reads in {output_R1}, but found {count_reads_in_fastq(output_R1)}"
                    kma_output = run_kma_quantification(kma_index, output_R1, output_R2, output_for_sample)
                    kallisto_output = run_kallisto_quantification(kallisto_index, output_R1, output_R2, output_for_sample)
                else:
                    expA, expB = get_number_of_reads_to_sample(R1_A, R1_B, proportion_A)
                # parse results
                kallisto_df = parse_kallisto_output(os.path.join(output_for_sample, "abundance.tsv"))
                kma_df = parse_kma_output(f"{output_for_sample}.mapstat")
                if dataset == "mosquito":
                    kallisto_df = sum_across_species(kallisto_df)
                    kma_df = sum_across_species(kma_df)
                # compute log error for each method
                if dataset == "mosquito":
                    A_target = sample_A
                    B_target = sample_B
                else:
                    A_target = ref_dict[sample_A]["original_id"]
                    B_target = ref_dict[sample_B]["original_id"]
                kma_match_A = kma_df.filter(pl.col("reference") == A_target)["kma_read_count"]
                kma_match_B = kma_df.filter(pl.col("reference") == B_target)["kma_read_count"]
                kallisto_match_A = kallisto_df.filter(pl.col("reference") == A_target)["kallisto_estimated_counts"]
                kallisto_match_B = kallisto_df.filter(pl.col("reference") == B_target)["kallisto_estimated_counts"]
                kma_count_A = kma_match_A.item()/2 if len(kma_match_A) > 0 else 0
                kma_count_B = kma_match_B.item()/2 if len(kma_match_B) > 0 else 0
                kallisto_count_A = kallisto_match_A.item() if len(kallisto_match_A) > 0 else 0
                kallisto_count_B = kallisto_match_B.item() if len(kallisto_match_B) > 0 else 0
                kma_total_assigned = kma_df["kma_read_count"].sum()/2
                kallisto_total_estimated = kallisto_df["kallisto_estimated_counts"].sum()
                kallisto_log_error = compute_log_error(expA, expB, kallisto_count_A, kallisto_count_B)
                kma_log_error = compute_log_error(expA, expB, kma_count_A, kma_count_B)
                results.append({
                    "dataset": dataset,
                    "sample_A": sample_A,
                    "sample_B": sample_B,
                    "proportion_A": proportion_A,
                    "expected_count_A": expA,
                    "expected_count_B": expB,
                    "kma_count_A": kma_count_A,
                    "kma_count_B": kma_count_B,
                    "kma_log_error": kma_log_error,
                    "kallisto_count_A": kallisto_count_A,
                    "kallisto_count_B": kallisto_count_B,
                    "kallisto_log_error": kallisto_log_error,
                    "kma_total_assigned": kma_total_assigned,
                    "kallisto_total_estimated": kallisto_total_estimated
                })
                # remove the mixture fastq files to save space
                if os.path.exists(output_R1) and os.path.exists(output_R2):
                    subprocess.run(f"rm {output_R1} {output_R2}", shell=True, executable="/bin/bash", check=True)
    return pl.DataFrame(results)