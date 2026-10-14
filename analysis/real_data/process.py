import numpy as np
import json
from Bio import SeqIO
from ..shared.kmers import get_kmer_list_from_fastq
from ..shared.mash import compute_jaccard_index, compute_mash_distance, get_kmers_from_sequence
from ..shared.mosquito_ids import return_species_from_target_id
from ..simulations.process import (
    build_kma_index,
    run_kma_quantification,
    build_kallisto_index,
    run_kallisto_quantification,
    parse_kma_output,
    parse_kallisto_output,
)
import polars as pl
import gzip
import os

def short_genus(species_name: str) -> str:
    if "Culex" in species_name:
        return species_name.replace("Culex", "Cx.")
    elif "Anopheles" in species_name:
        return species_name.replace("Anopheles", "An.")
    elif "Aedes" in species_name:
        return species_name.replace("Aedes", "Ae.")
    elif "Mansonia" in species_name:
        return species_name.replace("Mansonia", "Ma.")
    elif "Ochlerotatus" in species_name:
        return species_name.replace("Ochlerotatus", "Ae.")
    elif "Haemagogus" in species_name:
        return species_name.replace("Haemagogus", "Hg.")
    elif "Toxorhynchites" in species_name:
        return species_name.replace("Toxorhynchites", "Tx.")
    elif "Coquillettidia" in species_name:
        return species_name.replace("Coquillettidia", "Cq.")
    elif "Psorophora" in species_name:
        return species_name.replace("Psorophora", "Ps.")
    elif "Sabethes" in species_name:
        return species_name.replace("Sabethes", "Sa.")
    elif "Tripteroides" in species_name:
        return species_name.replace("Tripteroides", "Tr.")
    elif "Uranotaenia" in species_name:
        return species_name.replace("Uranotaenia", "Ur.")
    elif "Verallina" in species_name:
        return species_name.replace("Verallina", "Ve.")
    else:
        return species_name

def load_json(json_file: str) -> dict:
    with open(json_file) as f:
        data = json.load(f)
    return data

def parse_fasta_file(fasta_file: str) -> dict:
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences

def filter_seqs_by_json(fasta_file: str, json_file: str, json_key: str) -> dict:
    seqs = parse_fasta_file(fasta_file)
    data = load_json(json_file)
    selected_ids = set(data["target_sets"][json_key])
    # filter the sequences to only those in selected_ids
    filtered_seqs = {id: seq for id, seq in seqs.items() if id in selected_ids}
    return filtered_seqs

def report_mash_matrix_stats(matrix: np.ndarray, labels: list):
    min_mash = np.min(matrix[np.triu_indices_from(matrix, k=1)])  # Exclude diagonal
    max_mash = np.max(matrix[np.triu_indices_from(matrix, k=1)]) 
    mean_mash = np.mean(matrix[np.triu_indices_from(matrix, k=1)])
    print(f"Minimum MASH distance: {min_mash:.4f}")
    print(f"Maximum MASH distance: {max_mash:.4f}")
    print(f"Mean MASH distance: {mean_mash:.4f}")

def make_dict_of_refs(fasta_list: list) -> dict:
    ref_dict = {}
    for fasta_file in fasta_list:
        sequences = parse_fasta_file(fasta_file)
        for original_id, seq in sequences.items():
            new_id = original_id
            ref_type = "unknown"
            # Try to identify as mosquito
            species_name = return_species_from_target_id(original_id)
            if species_name is not None and species_name != "unknown":
                new_id = species_name.replace(" ", "_")
                ref_type = "mosquito"
            # Try to identify as COVID
            elif "Severe_acute_respiratory_syndrome_coronavirus" in original_id:
                new_id = original_id.split("_")[-1]
                ref_type = "SARS-Cov2"
            
            ref_dict[new_id] = {
                "original_id": original_id,
                "seq": seq,
                "type": ref_type
            }
    return ref_dict

def compute_mash_dists_between_samples_and_correct_ref(fasta_files: list, sample_config_json: str, kmer_size: int, tmp_dir: str) -> pl.DataFrame:
    results = []
    ref_dict = make_dict_of_refs(fasta_files)
    with open(sample_config_json) as f:
        config = json.load(f)
    for dataset in config:
        for sample in config[dataset]["samples"]:
            ref_seq = ref_dict[sample]["seq"]
            ref_kmers = get_kmers_from_sequence(ref_seq, kmer_size)
            R1 = config[dataset]["samples"][sample]["R1"]
            R2 = config[dataset]["samples"][sample]["R2"]
            sample_kmers = set(get_kmer_list_from_fastq([R1, R2], kmer_size, output_prefix=f"{tmp_dir}/{sample}_temp_kmers", threshold=1))
            jaccard_idx = compute_jaccard_index(ref_kmers, sample_kmers)
            mash_dist = compute_mash_distance(jaccard_idx, kmer_size)
            results.append({
                "sample": sample,
                "ref_seq": ref_seq,
                "fastq_R1": R1,
                "fastq_R2": R2,
                "n_ref_kmers": len(ref_kmers),
                "n_sample_kmers": len(sample_kmers),
                "jaccard_index": jaccard_idx,
                "mash_distance": mash_dist
            })
    return pl.DataFrame(results)

def count_reads_in_fastq(fastq_file: str) -> int:
    with gzip.open(fastq_file, 'rt') as f:
        return sum(1 for line in f) // 4

def run_kallisto_and_kma_id_on_data(fasta_files: list, sample_config_json: str, tmp_dir: str) -> pl.DataFrame:
    results = []
    ref_dict = make_dict_of_refs(fasta_files)
    # build indexes for each fasta file
    for i, fasta_file in enumerate(fasta_files):
        if i == 0:
            index_type = "mosquito"
        else:
            index_type = "SARS-Cov2"
        index_prefix = f"{tmp_dir}/{index_type}_index"
        build_kma_index(fasta_file, index_prefix)
        build_kallisto_index(fasta_file, index_prefix)
    with open(sample_config_json) as f:
        config = json.load(f)
    for dataset in config:
        index_prefix = f"{tmp_dir}/{dataset}_index"
        for sample in config[dataset]["samples"]:
            full_ref_id = ref_dict[sample]["original_id"]
            # run kma and kallisto
            output_for_sample = f"{tmp_dir}/{sample}_kma_kallisto_output"
            # if outputs don't exist, run the quantification, otherwise skip to parsing outputs
            if not os.path.exists(f"{output_for_sample}.mapstat"):
                kma_output = run_kma_quantification(index_prefix, config[dataset]["samples"][sample]["R1"], config[dataset]["samples"][sample]["R2"], output_for_sample)
            if not os.path.exists(f"{output_for_sample}/abundance.tsv"):
                kallisto_output = run_kallisto_quantification(index_prefix, config[dataset]["samples"][sample]["R1"], config[dataset]["samples"][sample]["R2"], output_for_sample)
            # parse outputs
            kallisto_df = parse_kallisto_output(os.path.join(output_for_sample, "abundance.tsv"))
            kma_df = parse_kma_output(f"{output_for_sample}.mapstat")
            # if dataset is mosquito, convert reference ids to species names
            if dataset == "mosquito":
                kallisto_df = kallisto_df.with_columns(
                    pl.col("reference").map_elements(lambda x: return_species_from_target_id(x).replace(" ", "_") if return_species_from_target_id(x) is not None else x).alias("reference")
                )
                kma_df = kma_df.with_columns(
                    pl.col("reference").map_elements(lambda x: return_species_from_target_id(x).replace(" ", "_") if return_species_from_target_id(x) is not None else x).alias("reference")
                )
                # sum counts for references that map to the same species
                kallisto_df = kallisto_df.with_columns(
                    pl.col("kallisto_estimated_counts").sum().over("reference").alias("kallisto_estimated_counts")
                )
                # if sample is Mansonia_uniformis, prnt where kallisto_df ref is Mansonia_uniformis and kma_df ref is Mansonia_uniformis, to check they match
                kma_df = kma_df.with_columns(
                    pl.col("kma_read_count").sum().over("reference").alias("kma_read_count")
                )
                kallisto_df = kallisto_df.unique(subset="reference")
                kma_df = kma_df.unique(subset="reference")
                full_ref_id = sample
            # extract counts of reads on correct reference
            kma_correct_ref_count = kma_df.filter(pl.col("reference") == full_ref_id)["kma_read_count"].sum()/2
            kma_off_target_count = kma_df.filter(pl.col("reference") != full_ref_id)["kma_read_count"].sum()/2
            kma_total_read_pairs_in_fastq = count_reads_in_fastq(config[dataset]["samples"][sample]["R1"])
            kallisto_correct_ref_count = kallisto_df.filter(pl.col("reference") == full_ref_id)["kallisto_estimated_counts"].sum()
            kallisto_off_target_count = kallisto_df.filter(pl.col("reference") != full_ref_id)["kallisto_estimated_counts"].sum()
            kallisto_total_read_pairs_in_fastq = count_reads_in_fastq(config[dataset]["samples"][sample]["R1"])
            results.append({
                "sample": sample,
                "dataset": dataset,
                "full_ref_id": full_ref_id,
                "kma_correct_ref_count": kma_correct_ref_count,
                "kma_off_target_count": kma_off_target_count,
                "kma_total_read_pairs_in_fastq": kma_total_read_pairs_in_fastq,
                "kallisto_correct_ref_count": kallisto_correct_ref_count,
                "kallisto_off_target_count": kallisto_off_target_count,
                "kallisto_total_read_pairs_in_fastq": kallisto_total_read_pairs_in_fastq
            })
    return pl.DataFrame(results)
            
def merge_mash_and_abundance_results(pairwise_mash_df: pl.DataFrame, mash_to_ref_df: pl.DataFrame, abundance_df: pl.DataFrame) -> pl.DataFrame:
    # join mash distance of df2 to df on sample name
    abundance_df = abundance_df.join(mash_to_ref_df.select(["sample", "mash_distance"]), on="sample", how="left")
    abundance_df = abundance_df.rename({"mash_distance": "mash_distance_to_correct_ref"})
    # if dataset is covid, split label1 and label2 in pairwise_mash_df by "_" and take end part
    pairwise_mash_df = pairwise_mash_df.with_columns(
        pl.when(pl.col("label1").str.contains("Severe_acute")).then(pl.col("label1").str.split("_").list.get(-1)).otherwise(pl.col("label1")).alias("label1"),
        pl.when(pl.col("label2").str.contains("Severe_acute")).then(pl.col("label2").str.split("_").list.get(-1)).otherwise(pl.col("label2")).alias("label2")
    )
    # now for each sample in abundance_df, find min mash distance to any reference in pairwise_mash_df, and add that to abundance_df as a new column
    min_mash_to_any_ref = {}
    for sample in abundance_df["sample"].to_list():
        mash_distances = pairwise_mash_df.filter(
            (pl.col("label1") == sample) | (pl.col("label2") == sample)
        )["mash_distance"].to_list()
        min_mash_to_any_ref[sample] = min(mash_distances)
    min_mash_lookup = pl.DataFrame(
        {
            "sample": list(min_mash_to_any_ref.keys()),
            "min_mash_distance_to_any_ref": list(min_mash_to_any_ref.values()),
        }
    )
    abundance_df = abundance_df.join(min_mash_lookup, on="sample", how="left")
    # get proportions in table for plotting
    abundance_df = abundance_df.with_columns(
            (pl.col("kma_total_read_pairs_in_fastq") - pl.col("kma_correct_ref_count") - pl.col("kma_off_target_count")).alias("kma_unassigned_count"),
            (pl.col("kallisto_total_read_pairs_in_fastq") - pl.col("kallisto_correct_ref_count") - pl.col("kallisto_off_target_count")).alias("kallisto_unassigned_count")
        ).with_columns(
            (pl.col("kma_correct_ref_count") / pl.col("kma_total_read_pairs_in_fastq")).alias("kma_correct_ref_prop"),
            (pl.col("kma_off_target_count") / pl.col("kma_total_read_pairs_in_fastq")).alias("kma_off_target_prop"),
            (pl.col("kma_unassigned_count") / pl.col("kma_total_read_pairs_in_fastq")).alias("kma_unassigned_prop"),
            (pl.col("kallisto_correct_ref_count") / pl.col("kallisto_total_read_pairs_in_fastq")).alias("kallisto_correct_ref_prop"),
            (pl.col("kallisto_off_target_count") / pl.col("kallisto_total_read_pairs_in_fastq")).alias("kallisto_off_target_prop"),
            (pl.col("kallisto_unassigned_count") / pl.col("kallisto_total_read_pairs_in_fastq")).alias("kallisto_unassigned_prop")
        ).with_columns(
            (1 - pl.col("kma_correct_ref_prop")).alias("kma_not_assigned_to_correct_ref_prop"),
            (1 - pl.col("kallisto_correct_ref_prop")).alias("kallisto_not_assigned_to_correct_ref_prop")
        )
    return abundance_df
