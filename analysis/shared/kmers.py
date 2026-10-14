import subprocess
import os
import polars as pl

def count_kmers_in_fastq_with_jellyfish(fastq_file1: str, fasta_file2: str, kmer_size: int, output_prefix: str, threshold: int) -> None:
    # if output file already exists, skip counting
    if os.path.exists(f"{output_prefix}_{kmer_size}_kmers.txt"):
        print(f"Output file {output_prefix}_{kmer_size}_kmers.txt already exists, skipping kmer counting for {fastq_file1}")
        return
    # Count kmers using jellyfish
    command = f"jellyfish count -m {kmer_size} -s 500M -t 4 -C -o {output_prefix}.jf <(gzcat {fastq_file1}) <(gzcat {fasta_file2})"
    subprocess.run(command, shell=True, executable="/bin/bash", check=True)
    # Dump the kmer counts to a text file
    command_dump = f"jellyfish dump -L {threshold + 1} {output_prefix}.jf_0 > {output_prefix}_{kmer_size}_kmers.txt"
    subprocess.run(command_dump, shell=True, executable="/bin/bash", check=True)
    # filter .jf file to save space
    if os.path.exists(f"{output_prefix}.jf_0"):
        os.remove(f"{output_prefix}.jf_0")

def get_kmer_list_from_fastq(fastq_files: list, kmer_size: int, output_prefix: str, threshold: int) -> list:
    count_kmers_in_fastq_with_jellyfish(fastq_files[0], fastq_files[1], kmer_size, output_prefix, threshold)
    kmer_counts_df = make_kmer_count_table(f"{output_prefix}_{kmer_size}_kmers.txt")
    kmers = kmer_counts_df["kmer"].to_list()
    # make kmers uppercase
    kmers = [kmer.upper() for kmer in kmers]
    return kmers

def make_kmer_count_table(input_file: str) -> pl.DataFrame:
    records = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                kmer = next(f).strip()
                count = int(line.strip().replace('>', ''))
                records.append((kmer, count))
    return pl.DataFrame(records, schema=["kmer", "count"])