import pandas as pd


def create_no_knowledge_data(t2_annotations, t1_bp, t1_mf, t1_cc, output_file):
    """ Creates file with protein ids that are part of the no-knowledge data. The no-knowledge data
    consists of proteins that appear (are annotated) in t2 annotation file and did not appear in any
    of the other ontologies in t1.
        - t2 annotations = current annotations
        - t1 annotations = previous annotations

    :param t2_annotations: current annotations for some gene ontology type
    :type t2_annotations: dict
    :param t1_bp: previous annotations for BP gene ontology
    :type t1_bp: dict
    :param t1_mf: previous annotations for MF gene ontology
    :type t1_mf: dict
    :param t1_cc: previous annotations for CC gene ontology
    :type t1_cc: dict
    :param output_file: file path to save the results
    :type output_file: str
    :return: None
    """
    with open(output_file, 'w') as f:
        f.write('protein_id\n')
        for protein in t2_annotations.keys():
            if protein not in t1_bp and protein not in t1_mf and protein not in t1_cc:
                f.write(protein + '\n')


def create_limited_knowledge_data(t2_annotations, t1_annotations, t1_annotations2, t1_annotations3, output_file):
    """ Creates file with protein ids that are part of the limited-knowledge data. The limited-knowledge
    data consists of proteins that appear (are annotated) in t2 annotation file for given ontology and did
    not appear in the same ontology, but appear in at least one of the other ontologies in t1.
        - t2 annotations = current annotations
        - t1 annotations = previous annotations

    :param t2_annotations: current annotations for the given gene ontology type
    :type t2_annotations: dict
    :param t1_annotations: previous annotations for the given gene ontology type
    :type t1_annotations: dict
    :param t1_annotations2: previous annotations for the second gene ontology type
    :type t1_annotations2: dict
    :param t1_annotations3: previous annotations for the third gene ontology type
    :type t1_annotations3: dict
    :param output_file: file path to save the results
    :type output_file: str
    :return: None
    """
    with open(output_file, 'w') as f:
        f.write('protein_id\n')
        for protein in t2_annotations.keys():
            if protein not in t1_annotations and (protein in t1_annotations2 or protein in t1_annotations3):
                f.write(protein + '\n')


def create_knowledge_data(t1, t2, output_file):
    """ Creates file with protein ids that are part of the knowledge data. The knowledge data consists
    of proteins that appear (are annotated) in t2 annotation file for given ontology and appear in the
    same ontology in t1.
        - t2 annotations = current annotations
        - t1 annotations = previous annotations

    :param t1: previous annotations for the given gene ontology type
    :type t1: dict
    :param t2: current annotations for the given gene ontology type
    :type t2: dict
    :param output_file: file path to save the results
    :type output_file: str
    :return: None
    """
    with open(output_file, 'w') as f:
        f.write('protein_id\n')
        for protein in t1.keys():
            if protein in t2 and set(t1[protein]) != set(t2[protein]):
                f.write(protein + '\n')


def create_benchmark_data(type_filtering):
    t1_bp_file = f'data/human_ppi_{type_filtering}/HumanPPI_GO_BP_no_bias.txt'
    t1_mf_file = f'data/human_ppi_{type_filtering}/HumanPPI_GO_MF_no_bias.txt'
    t1_cc_file = f'data/human_ppi_{type_filtering}/HumanPPI_GO_CC_no_bias.txt'
    t2_bp_file = f'data/human_ppi_{type_filtering}/t2/HumanPPI_GO_BP_no_bias.txt'
    t2_mf_file = f'data/human_ppi_{type_filtering}/t2/HumanPPI_GO_MF_no_bias.txt'
    t2_cc_file = f'data/human_ppi_{type_filtering}/t2/HumanPPI_GO_CC_no_bias.txt'
    t1_bp = pd.read_table(t1_bp_file, header=0, names=['PID', 'GO']).groupby('PID')['GO'].apply(list).to_dict()
    bp_anno = set([x for y in list(t1_bp.values()) for x in y])
    t2 = pd.read_table(t2_bp_file, header=0, names=['PID', 'GO'])
    t2 = t2[t2.GO.isin(bp_anno)]
    t2_bp = t2.groupby('PID')['GO'].apply(list).to_dict()
    t1_mf = pd.read_table(t1_mf_file, header=0, names=['PID', 'GO']).groupby('PID')['GO'].apply(list).to_dict()
    mf_anno = set([x for y in list(t1_mf.values()) for x in y])
    t2 = pd.read_table(t2_mf_file, header=0, names=['PID', 'GO'])
    t2 = t2[t2.GO.isin(mf_anno)]
    t2_mf = t2.groupby('PID')['GO'].apply(list).to_dict()
    t1_cc = pd.read_table(t1_cc_file, header=0, names=['PID', 'GO']).groupby('PID')['GO'].apply(list).to_dict()
    cc_anno = set([x for y in list(t1_cc.values()) for x in y])
    t2 = pd.read_table(t2_cc_file, header=0, names=['PID', 'GO'])
    t2 = t2[t2.GO.isin(cc_anno)]
    t2_cc = t2.groupby('PID')['GO'].apply(list).to_dict()

    create_no_knowledge_data(t2_bp, t1_bp, t1_mf, t1_cc, f'data/final/human_ppi_{type_filtering}/BP_no_bias_NK.txt')
    create_no_knowledge_data(t2_mf, t1_bp, t1_mf, t1_cc, f'data/final/human_ppi_{type_filtering}/MF_no_bias_NK.txt')
    create_no_knowledge_data(t2_cc, t1_bp, t1_mf, t1_cc, f'data/final/human_ppi_{type_filtering}/CC_no_bias_NK.txt')

    create_limited_knowledge_data(t2_bp, t1_bp, t1_mf, t1_cc,
                                  f'data/final/human_ppi_{type_filtering}/BP_no_bias_LK.txt')
    create_limited_knowledge_data(t2_mf, t1_mf, t1_bp, t1_cc,
                                  f'data/final/human_ppi_{type_filtering}/MF_no_bias_LK.txt')
    create_limited_knowledge_data(t2_cc, t1_cc, t1_bp, t1_mf,
                                  f'data/final/human_ppi_{type_filtering}/CC_no_bias_LK.txt')

    create_knowledge_data(t1_bp, t2_bp, f'data/final/human_ppi_{type_filtering}/BP_no_bias_K.txt')
    create_knowledge_data(t1_mf, t2_mf, f'data/final/human_ppi_{type_filtering}/MF_no_bias_K.txt')
    create_knowledge_data(t1_cc, t2_cc, f'data/final/human_ppi_{type_filtering}/CC_no_bias_K.txt')


if __name__ == '__main__':
    create_benchmark_data('700')
    create_benchmark_data('900')
