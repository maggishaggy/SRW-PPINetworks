"""
Step 3
This file divides the files HumanPPI_GO (from both 700 and 900 interactions) into
three sets according to MF, BP and CC terms
"""
import time
import pandas as pd


def divide_annotations(file_annotations, output_file_bp, output_file_mf, output_file_cc):
    """ Divides protein go annotations into three sets for MF, BP and CC go terms

    :param file_annotations: file path to file containing the protein go annotations
    :type file_annotations: str
    :param output_file_bp: file path to file for saving the BP protein go annotations
    :type output_file_bp: str
    :param output_file_mf: file path to file for saving the MF protein go annotations
    :type output_file_mf: str
    :param output_file_cc: file path to file for saving the CC protein go annotations
    :type output_file_cc: str
    :return: None
    """
    go = pd.read_csv('../data/go/BPGOterms.txt', header=None, sep=' ', names=['term'])
    go_terms_bp = go['term'].values.tolist()
    go = pd.read_csv('../data/go/MFGOterms.txt', header=None, sep=' ', names=['term'])
    go_terms_mf = go['term'].values.tolist()
    go = pd.read_csv('../data/go/CCGOterms.txt', header=None, sep=' ', names=['term'])
    go_terms_cc = go['term'].values.tolist()
    del go
    annotations = pd.read_csv(file_annotations, sep='\t', header=0)
    bp_annotations = annotations.join(pd.DataFrame({'go_id': go_terms_bp}).set_index('go_id'), how='inner', on='go_id')
    bp_annotations.to_csv(output_file_bp, sep='\t', index=False)
    del bp_annotations
    mf_annotations = annotations.join(pd.DataFrame({'go_id': go_terms_mf}).set_index('go_id'), how='inner', on='go_id')
    mf_annotations.to_csv(output_file_mf, sep='\t', index=False)
    del mf_annotations
    cc_annotations = annotations.join(pd.DataFrame({'go_id': go_terms_cc}).set_index('go_id'), how='inner', on='go_id')
    cc_annotations.to_csv(output_file_cc, sep='\t', index=False)
    del cc_annotations


if __name__ == '__main__':
    print("(" + time.strftime("%c") + ")  Dividing the HumanPPI700GO file...")
    divide_annotations('../data/human_ppi_700/HumanPPI_GO.txt',
                       '../data/human_ppi_700/HumanPPI_GO_BP.txt',
                       '../data/human_ppi_700/HumanPPI_GO_MF.txt',
                       '../data/human_ppi_700/HumanPPI_GO_CC.txt')
    print("(" + time.strftime("%c") + ")  Dividing the HumanPPI900GO file...")
    divide_annotations('../data/human_ppi_900/HumanPPI_GO.txt',
                       '../data/human_ppi_900/HumanPPI_GO_BP.txt',
                       '../data/human_ppi_900/HumanPPI_GO_MF.txt',
                       '../data/human_ppi_900/HumanPPI_GO_CC.txt')
