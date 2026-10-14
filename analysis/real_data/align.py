import subprocess
import json
import os
from copy import deepcopy

def align_fastq_to_fasta_with_minimap2_to_bam(fasta_file: str, R1: str, R2: str, output_file: str) -> None:
    command = f"minimap2 -ax sr --secondary=yes {fasta_file} {R1} {R2} | samtools view -F4 -bS - | samtools sort -n -o {output_file} -"
    subprocess.run(command, shell=True, executable="/bin/bash", check=True)

def extract_aligned_reads_from_bam(bam_file: str, output_fastq1: str, output_fastq2: str) -> None:
    # Keep paired reads where both mates are mapped, without requiring the proper-pair flag.
    command = f"samtools view -b {bam_file} | bedtools bamtofastq -i - -fq {output_fastq1.replace('.gz', '')} -fq2 {output_fastq2.replace('.gz', '')}"
    subprocess.run(command, shell=True, executable="/bin/bash", check=True)
    # gzip the output fastq files
    subprocess.run(f"gzip {output_fastq1.replace('.gz', '')}", shell=True, executable="/bin/bash", check=True)
    subprocess.run(f"gzip {output_fastq2.replace('.gz', '')}", shell=True, executable="/bin/bash", check=True)

def align_covid_samples_to_refs(sample_json: str, ref_fasta: str, output_dir: str = "results/align") -> None:
    with open(sample_json, 'r') as f:
        config = json.load(f)
    updated_config = deepcopy(config)
    os.makedirs(output_dir, exist_ok=True)
    for sample in config["SARS-Cov2"]["samples"]:
        R1 = config["SARS-Cov2"]["samples"][sample]["R1"]
        R2 = config["SARS-Cov2"]["samples"][sample]["R2"]
        output_file = f"{output_dir}/{sample}_aligned.bam"
        output_fastq1 = f"{output_dir}/{sample}_aligned_R1.fastq.gz"
        output_fastq2 = f"{output_dir}/{sample}_aligned_R2.fastq.gz"
        # if output fastqs don't exist, run alignment and extraction
        if not (os.path.exists(output_fastq1) and os.path.exists(output_fastq2)):
            align_fastq_to_fasta_with_minimap2_to_bam(ref_fasta, R1, R2, output_file)
            extract_aligned_reads_from_bam(output_file, output_fastq1, output_fastq2)
            # delete bam file to save space
            subprocess.run(f"rm {output_file}", shell=True, executable="/bin/bash", check=True)
        # update the config with the paths to the aligned fastq files
        updated_config["SARS-Cov2"]["samples"][sample]["original_R1"] = R1
        updated_config["SARS-Cov2"]["samples"][sample]["original_R2"] = R2
        updated_config["SARS-Cov2"]["samples"][sample]["R1"] = output_fastq1
        updated_config["SARS-Cov2"]["samples"][sample]["R2"] = output_fastq2
    # save the updated config to a new json file
    with open(sample_json.replace(".json", ".updated.json"), 'w') as f:
        json.dump(updated_config, f, indent=4)