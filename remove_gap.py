import os
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
import argparse

# input and output dir

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='test_align', required=False)
    parser.add_argument('--output_folder', type=str, default='.test_align_no_gap', required=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder

    # if ptah does not exist
    os.makedirs(output_folder, exist_ok=True)


    # count = 0
    for filename in os.listdir(input_folder):
        if filename.endswith(".fasta") or filename.endswith(".fasta"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # real alignment
            alignment = AlignIO.read(input_path, "fasta")

            # read all alignments without gaps
            ungapped_columns = []
            for i in range(alignment.get_alignment_length()):
                column = alignment[:, i]
                if '-' not in column and '.' not in column and 'N' not in column:
                    ungapped_columns.append(i)

            # re-construct alignment
            new_records = []
            for record in alignment:
                new_seq = ''.join(record.seq[i] for i in ungapped_columns)
                record.seq = record.seq.__class__(new_seq)
                new_records.append(record)

            new_alignment = MultipleSeqAlignment(new_records)

            # save alignment
            AlignIO.write(new_alignment, output_path, "fasta")
