from typing import List
import argparse

aparser = argparse.ArgumentParser(prog='idlppbopt')
aparser.add_argument(
    '-i', '--infile', type=str, required=True, help='Input file in CSV format.')
aparser.add_argument(
    '-o', '--outfile', type=str, default='-', required=False, help=(
		'(opt.) Output file in CSV format. If not specified, output will be '
        'written to STDOUT.'))
aparser.add_argument(
    '--smiles-column', type=str, default=None, required=False, help=(
        '(opt.) Column name of the SMILES data.'))
aparser.add_argument(
    '---output-column', type=str, default='IDL-PPBopt', required=False, help=(
        "(opt.) Column to which the output is written (default: 'IDL-PPBopt')"))


def parse_arguments(arguments: List[str]):
    args = aparser.parse_args(arguments)
    return args